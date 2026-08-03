"""
油位表检测推理脚本 - 本地测试用

用法：
  # 单张图片
  python test.py --image test.jpg

  # 整个文件夹
  python test.py --image ./test_images/

  # 指定输出目录
  python test.py --image test.jpg --output ./results/

  # 调整置信度阈值
  python test.py --image test.jpg --gauge-conf 0.3 --redline-conf 0.25
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from oil_level_gauge_algorithm_yolo import OilLevelGaugeDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="油位表油位识别推理脚本")
    parser.add_argument(
        "--image", type=str, required=True,
        help="输入图片路径或文件夹路径",
    )
    parser.add_argument(
        "--gauge-model", type=str, default="level_best.pt",
        help="油位表检测模型路径（默认: level_best.pt）",
    )
    parser.add_argument(
        "--redline-model", type=str, default="red_best.pt",
        help="红线检测模型路径（默认: best.pt）",
    )
    parser.add_argument(
        "--output", type=str, default="./results",
        help="输出目录（默认: ./results）",
    )
    parser.add_argument(
        "--gauge-conf", type=float, default=0.25,
        help="油位表检测置信度阈值（默认: 0.25）",
    )
    parser.add_argument(
        "--gauge-iou", type=float, default=0.45,
        help="油位表检测 NMS IoU 阈值（默认: 0.45）",
    )
    parser.add_argument(
        "--redline-conf", type=float, default=0.25,
        help="红线检测置信度阈值（默认: 0.25）",
    )
    parser.add_argument(
        "--redline-iou", type=float, default=0.45,
        help="红线检测 NMS IoU 阈值（默认: 0.45）",
    )
    parser.add_argument(
        "--no-verbose", action="store_true",
        help="不打印详细推理信息",
    )
    return parser.parse_args()


def collect_images(path: str) -> list[Path]:
    """收集图片文件列表。"""
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        return sorted([f for f in p.iterdir() if f.suffix.lower() in exts])
    raise FileNotFoundError(f"路径不存在: {path}")


def run_single(
    detector: OilLevelGaugeDetector,
    image_path: Path,
    output_dir: Path,
    options: dict,
    verbose: bool = True,
) -> dict:
    """对单张图片执行推理并保存结果。"""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  [SKIP] 无法读取图片: {image_path}")
        return {}

    result = detector.infer(image, params=options)

    # 保存结果图片
    result_img = result.get("result_image")
    if result_img is not None:
        save_path = output_dir / image_path.name
        cv2.imwrite(str(save_path), result_img)

    # 打印推理信息
    if verbose:
        state = result.get("state", "unknown")
        cost = result.get("cost_ms", 0)
        reason = result.get("reason", "")
        conf = result.get("confidence", 0.0)
        gauge_bbox = result.get("gauge_bbox", [])
        ref_lines = result.get("reference_lines_y", [])
        oil_y = result.get("oil_level_y")

        state_icon = {"normal": "NORMAL", "abnormal": "ABNORMAL", "unknown": "UNKNOWN"}.get(state, "?")
        print(f"  状态: {state_icon}")
        print(f"  耗时: {cost} ms")
        print(f"  原因: {reason}")
        redline_method = result.get("redline_method", "N/A")
        method_label = {"yolo": "YOLO 模型", "hsv_fallback": "传统 HSV (回退)"}.get(redline_method, redline_method)
        print(f"  红线检测方法: {method_label}")
        print(f"  油位表置信度: {conf:.4f}")
        if gauge_bbox:
            print(f"  油位表框: x={gauge_bbox[0]} y={gauge_bbox[1]} w={gauge_bbox[2]} h={gauge_bbox[3]}")
        if ref_lines:
            print(f"  红线 Y 坐标: 上={ref_lines[0]} 下={ref_lines[1]}")
        else:
            print(f"  红线 Y 坐标: 未检测到")
        if oil_y is not None:
            ratio = result.get("oil_level_position_ratio")
            score = result.get("oil_level_score", 0)
            print(f"  油面 Y 坐标: {oil_y} (位置比={ratio}, 得分={score})")
        else:
            print(f"  油面 Y 坐标: 未检测到")

        print(f"  曝光过度: {result.get('is_overexposed', 'N/A')}")
        print(f"  偏色: {result.get('color_tint', 'N/A')}")
        print(f"  结果已保存: {output_dir / image_path.name}")

    return result


def main() -> None:
    args = parse_args()

    # 收集图片
    images = collect_images(args.image)
    if not images:
        print("未找到任何图片文件")
        return

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    print(f"加载油位表模型: {args.gauge_model}")
    print(f"加载红线模型:   {args.redline_model}")
    detector = OilLevelGaugeDetector(args.gauge_model)
    detector.load()
    detector.load_redline_model(args.redline_model)
    print("模型加载完成\n")

    # 算法参数
    options = {
        "det_conf": args.gauge_conf,
        "det_iou": args.gauge_iou,
        "redline_det_conf": args.redline_conf,
        "redline_det_iou": args.redline_iou,
    }

    # 逐张推理
    total = len(images)
    normal_count = 0
    abnormal_count = 0
    unknown_count = 0
    yolo_redline_count = 0
    hsv_fallback_count = 0
    total_time = 0.0

    print(f"共 {total} 张图片待处理\n{'='*50}")

    for i, img_path in enumerate(images, 1):
        print(f"\n[{i}/{total}] {img_path.name}")

        result = run_single(detector, img_path, output_dir, options, verbose=not args.no_verbose)

        if result:
            state = result.get("state", "unknown")
            if state == "normal":
                normal_count += 1
            elif state == "abnormal":
                abnormal_count += 1
            else:
                unknown_count += 1
            total_time += result.get("cost_ms", 0)
            if result.get("redline_method") == "yolo":
                yolo_redline_count += 1
            elif result.get("redline_method") == "hsv_fallback":
                hsv_fallback_count += 1

    # 统计汇总
    print(f"\n{'='*50}")
    print(f"推理完成!")
    print(f"  总计: {total} 张")
    print(f"  NORMAL:   {normal_count}")
    print(f"  ABNORMAL: {abnormal_count}")
    print(f"  UNKNOWN:  {unknown_count}")
    print(f"  红线检测: YOLO={yolo_redline_count} HSV回退={hsv_fallback_count}")
    if total > 0:
        print(f"  平均耗时: {total_time / total:.1f} ms")
    print(f"  结果目录: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
