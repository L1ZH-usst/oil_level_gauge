"""
油位表 YOLOv8 目标检测训练脚本

用法：
  python train.py                        # 使用默认参数训练
  python train.py --epochs 100           # 指定训练轮数
  python train.py --model yolov8m.pt     # 使用中等模型
  python train.py --resume               # 从上次中断处恢复训练
  python train.py --export               # 训练完成后导出 ONNX
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 解决 ultralytics 在某些环境下的 OpenMP 冲突问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO


# ===========================================================================
#  默认训练参数
# ===========================================================================

DEFAULT_CONFIG = {
    # === 模型 ===
    "model": "yolov8s.pt",          # 预训练模型：yolov8n/s/m/l/x，n 最快，x 最精

    # === 数据集 ===
    "data": "dataset/data.yaml",     # 数据集配置文件路径

    # === 训练超参数 ===
    "epochs": 150,                   # 训练总轮数
    "batch": -1,                     # 批次大小（根据 GPU 显存调整，8G 显存建议 8~16）
    "imgsz": 1024,                   # 输入图片尺寸（与数据集原图 1408x1024 接近）
    "patience": 30,                  # 早停耐心值：连续 N 轮 val/mAP 不提升则停止

    # === 优化器 ===
    "optimizer": "AdamW",            # 优化器：SGD / Adam / AdamW / NAdam / RAdam / RMSProp
    "lr0": 0.001,                    # 初始学习率
    "lrf": 0.01,                     # 最终学习率 = lr0 * lrf
    "momentum": 0.937,               # SGD 动量
    "weight_decay": 0.0005,          # 权重衰减
    "warmup_epochs": 3,              # 学习率预热轮数

    # === 数据增强 ===
    "hsv_h": 0.015,                  # 色调增强幅度
    "hsv_s": 0.4,                    # 饱和度增强幅度
    "hsv_v": 0.4,                    # 亮度增强幅度
    "degrees": 5.0,                  # 旋转角度范围（±度）
    "translate": 0.1,                # 平移范围（占图片比例）
    "scale": 0.3,                    # 缩放范围（±比例）
    "shear": 2.0,                    # 剪切角度范围
    "perspective": 0.0,              # 透视变换强度
    "flipud": 0.0,                   # 上下翻转概率（油位表不宜上下翻，保持 0）
    "fliplr": 0.5,                   # 左右翻转概率
    "mosaic": 1.0,                   # Mosaic 增强概率
    "mixup": 0.0,                    # MixUp 增强概率
    "copy_paste": 0.0,               # 复制粘贴增强概率

    # === 训练策略 ===
    "close_mosaic": 15,              # 最后 N 轮关闭 Mosaic（让模型适应真实分布）
    "amp": True,                     # 混合精度训练（加速且省显存）
    "cos_lr": True,                  # 使用余弦退火学习率调度

    # === 其他 ===
    "workers": 4,                    # 数据加载线程数
    "device": "",                    # 训练设备："" 自动选择, "0", "0,1", "cpu"
    "project": "runs/train",         # 输出目录
    "name": "oil_level_gauge",       # 实验名称
    "exist_ok": True,                # 覆盖已有同名实验目录
    "seed": 42,                      # 随机种子（可复现）
    "verbose": True,                 # 详细输出
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数，覆盖默认配置。"""
    parser = argparse.ArgumentParser(description="油位表 YOLOv8 训练脚本")

    parser.add_argument("--model", type=str, default=None,
                        help="预训练模型路径或名称（如 yolov8s.pt）")
    parser.add_argument("--data", type=str, default=None,
                        help="数据集配置文件路径（data.yaml）")
    parser.add_argument("--epochs", type=int, default=None,
                        help="训练轮数")
    parser.add_argument("--batch", type=int, default=None,
                        help="批次大小（-1 为 AutoBatch）")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="输入图片尺寸")
    parser.add_argument("--patience", type=int, default=None,
                        help="早停耐心值")
    parser.add_argument("--device", type=str, default=None,
                        help="训练设备（如 '0', '0,1', 'cpu'）")
    parser.add_argument("--lr0", type=float, default=None,
                        help="初始学习率")
    parser.add_argument("--optimizer", type=str, default=None,
                        help="优化器名称")
    parser.add_argument("--resume", action="store_true",
                        help="从上次中断处恢复训练")
    parser.add_argument("--export", action="store_true",
                        help="训练完成后导出 ONNX 模型")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="自定义预训练权重路径（如 weights/best.pt）")

    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    """
    执行 YOLOv8 训练。

    流程：
      1. 合并命令行参数和默认配置
      2. 加载预训练模型
      3. 调用 model.train() 开始训练
      4. （可选）导出 ONNX 模型
    """
    # 合并配置：命令行参数优先于默认值
    config = dict(DEFAULT_CONFIG)
    if args.model is not None:
        config["model"] = args.model
    if args.data is not None:
        config["data"] = args.data
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch is not None:
        config["batch"] = args.batch
    if args.imgsz is not None:
        config["imgsz"] = args.imgsz
    if args.patience is not None:
        config["patience"] = args.patience
    if args.device is not None:
        config["device"] = args.device
    if args.lr0 is not None:
        config["lr0"] = args.lr0
    if args.optimizer is not None:
        config["optimizer"] = args.optimizer
    if args.pretrained is not None:
        config["model"] = args.pretrained

    # 打印训练配置
    print("=" * 60)
    print("油位表 YOLOv8 训练")
    print("=" * 60)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 60)

    # 加载模型（自动下载预训练权重）
    model = YOLO(config["model"])

    # 恢复训练 or 从头训练
    if args.resume:
        print("\n>> 从上次中断处恢复训练...")
        model.train(resume=True)
    else:
        print(f"\n>> 开始训练 {config['epochs']} 轮...")
        model.train(
            data=config["data"],
            epochs=config["epochs"],
            batch=config["batch"],
            imgsz=config["imgsz"],
            patience=config["patience"],
            optimizer=config["optimizer"],
            lr0=config["lr0"],
            lrf=config["lrf"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
            warmup_epochs=config["warmup_epochs"],
            hsv_h=config["hsv_h"],
            hsv_s=config["hsv_s"],
            hsv_v=config["hsv_v"],
            degrees=config["degrees"],
            translate=config["translate"],
            scale=config["scale"],
            shear=config["shear"],
            perspective=config["perspective"],
            flipud=config["flipud"],
            fliplr=config["fliplr"],
            mosaic=config["mosaic"],
            mixup=config["mixup"],
            copy_paste=config["copy_paste"],
            close_mosaic=config["close_mosaic"],
            amp=config["amp"],
            cos_lr=config["cos_lr"],
            workers=config["workers"],
            device=config["device"],
            project=config["project"],
            name=config["name"],
            exist_ok=config["exist_ok"],
            seed=config["seed"],
            verbose=config["verbose"],
        )

    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"  最佳权重: runs/train/{config['name']}/weights/best.pt")
    print(f"  最终权重: runs/train/{config['name']}/weights/last.pt")
    print("=" * 60)

    # === 在验证集上评估最佳模型 ===
    best_weights = Path(config["project"]) / config["name"] / "weights" / "best.pt"
    if best_weights.exists():
        print("\n>> 使用最佳权重在验证集上评估...")
        best_model = YOLO(str(best_weights))
        metrics = best_model.val(data=config["data"], imgsz=config["imgsz"])
        print(f"\n  mAP50:      {metrics.box.map50:.4f}")
        print(f"  mAP50-95:   {metrics.box.map:.4f}")
        print(f"  Precision:  {metrics.box.mp:.4f}")
        print(f"  Recall:     {metrics.box.mr:.4f}")

    # === 可选：导出 ONNX ===
    if args.export and best_weights.exists():
        print("\n>> 导出 ONNX 模型...")
        export_model = YOLO(str(best_weights))
        export_model.export(format="onnx", imgsz=config["imgsz"], simplify=True)
        print(f"  ONNX 模型已保存到: {best_weights.with_suffix('.onnx')}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
