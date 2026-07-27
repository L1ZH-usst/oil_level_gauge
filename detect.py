"""
油位表本地检测推理脚本

用法：
  # 检测单张图片
  python detect.py --image dataset/val/images/3809_2026_07_22_16_22_26_865.jpg

  # 批量检测整个文件夹
  python detect.py --image dataset/val/images/

  # 指定模型权重和输出目录
  python detect.py --image dataset/val/images/ --model weights/best.pt --output results/

  # 调整检测置信度
  python detect.py --image dataset/val/images/ --conf 0.3

  # 只看结果不保存图片
  python detect.py --image dataset/val/images/ --no-save
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# 解决某些环境下的 OpenMP 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np

# 将脚本所在目录加入 Python 路径，确保能找到算法模块
sys.path.insert(0, str(Path(__file__).resolve().parent))
from oil_level_gauge_algorithm_annotated import OilLevelGaugeDetector


# 支持的图片格式
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="油位表本地检测推理脚本")

    parser.add_argument("--image", type=str, required=True,
                        help="待检测图片路径或文件夹路径")
    parser.add_argument("--model", type=str, default=r"C:\Users\elitedatai\Desktop\油位表识别\runs\detect\runs\train\oil_level_gauge\weights\best.pt",
                        help="YOLO 模型权重文件路径（默认: weights/best.pt）")
    parser.add_argument("--output", type=str, default="results",
                        help="结果图片输出目录（默认: results/）")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="YOLO 检测置信度阈值（默认: 0.25）")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU 阈值（默认: 0.45）")
    parser.add_argument("--no-save", action="store_true",
                        help="不保存结果图片（只在终端打印结果）")
    parser.add_argument("--no-show", action="store_true",
                        help="不弹窗显示结果图片")

    return parser.parse_args()


def collect_images(path: str) -> list[Path]:
    """收集指定路径下的所有图片文件。支持单文件或文件夹。"""
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        images = []
        for ext in IMAGE_EXTENSIONS:
            images.extend(p.glob(f"*{ext}"))
            images.extend(p.glob(f"*{ext.upper()}"))
        images.sort()
        return images
    print(f"错误: 路径不存在 {path}")
    sys.exit(1)


def print_result(filepath: str, result: dict) -> None:
    """在终端打印单张图片的检测结果。"""
    state = result["state"]
    reason = result["reason"]
    cost = result["cost_ms"]
    conf = result["confidence"]

    # 状态图标
    icon = {"normal": "[NORMAL]", "abnormal": "[ABNORMAL]", "unknown": "[UNKNOWN]"}
    tag = icon.get(state, "[???]")

    # 基本信息行
    print(f"  {tag}  {Path(filepath).name}")
    print(f"           reason={reason}  conf={conf:.3f}  cost={cost}ms")

    # 油位详情（仅检测成功时）
    if state != "unknown":
        oil_y = result.get("oil_level_y", "-")
        ratio = result.get("oil_level_position_ratio", "-")
        score = result.get("oil_level_score", 0)
        tube = result.get("tube_bounds", [])
        lines_y = result.get("reference_lines_y", [])
        print(f"           oil_y={oil_y}  ratio={ratio}  score={score:.2f}")
        if tube:
            print(f"           tube_bounds={tube}")
        if lines_y:
            print(f"           ref_lines_y={lines_y}")
    print()


def main():
    args = parse_args()

    # 收集图片
    images = collect_images(args.image)
    if not images:
        print("未找到任何图片文件。")
        sys.exit(1)

    print("=" * 60)
    print("油位表检测推理")
    print("=" * 60)
    print(f"  模型:   {args.model}")
    print(f"  图片:   {args.image} ({len(images)} 张)")
    print(f"  输出:   {args.output}")
    print(f"  conf:   {args.conf}")
    print(f"  iou:    {args.iou}")
    print("=" * 60)
    print()

    # 加载模型
    detector = OilLevelGaugeDetector(args.model)
    detector.load()

    # 创建输出目录
    output_dir = Path(args.output)
    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)

    # 统计计数器
    total = 0
    normal_count = 0
    abnormal_count = 0
    unknown_count = 0
    total_ms = 0.0

    # 逐张检测
    for img_path in images:
        total += 1

        # 读取图片
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [ERROR]  {img_path.name} — 图片读取失败")
            continue

        # 执行推理
        result = detector.infer(image, params={"det_conf": args.conf, "det_iou": args.iou})

        # 打印结果
        print_result(str(img_path), result)

        # 统计
        state = result["state"]
        if state == "normal":
            normal_count += 1
        elif state == "abnormal":
            abnormal_count += 1
        else:
            unknown_count += 1
        total_ms += result["cost_ms"]

        # 保存结果图
        if not args.no_save:
            result_image = result.get("result_image")
            if result_image is not None:
                out_name = f"{img_path.stem}_result.jpg"
                out_path = output_dir / out_name
                cv2.imwrite(str(out_path), result_image)

        # 弹窗显示（可选）
        if not args.no_show:
            result_image = result.get("result_image")
            if result_image is not None:
                # 缩放到合理尺寸显示
                h, w = result_image.shape[:2]
                scale = min(1.0, 960 / max(h, w))
                if scale < 1.0:
                    show_img = cv2.resize(result_image, (int(w * scale), int(h * scale)))
                else:
                    show_img = result_image
                cv2.imshow("Oil Level Gauge Detection", show_img)
                key = cv2.waitKey(0)
                if key == 27:  # ESC 键退出
                    break

    if not args.no_show:
        cv2.destroyAllWindows()

    # 打印汇总
    avg_ms = total_ms / max(total, 1)
    print("=" * 60)
    print("检测汇总")
    print("=" * 60)
    print(f"  总图片数:  {total}")
    print(f"  NORMAL:    {normal_count}")
    print(f"  ABNORMAL:  {abnormal_count}")
    print(f"  UNKNOWN:   {unknown_count}")
    print(f"  平均耗时:  {avg_ms:.1f} ms/张")
    print(f"  总耗时:    {total_ms:.0f} ms")
    if not args.no_save:
        print(f"  结果保存:  {output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
