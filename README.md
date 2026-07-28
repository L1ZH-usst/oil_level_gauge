# 油位表识别系统

基于 YOLOv8 + 传统视觉的两段式油位表油位识别系统。自动检测图片中的油位表，定位液柱管体、红色参考线和油面高度，判定油位状态是否正常。

## 算法流程

```
输入图片
  │
  ▼
┌─────────────────────┐
│ 第一段：YOLO 目标检测 │  ← best.pt (YOLOv8s)
│ 定位油位表区域 (bbox) │
└─────────┬───────────┘
          │ 裁剪 ROI
          ▼
┌─────────────────────────────────┐
│ 第二段：传统视觉规则判定          │
│                                 │
│  ① Sobel 梯度 → 管体左右边界    │
│  ② HSV 红色分割 → 上下参考线    │
│     (仅管体左边界左侧区域搜索)   │
│  ③ 亮度梯度 → 油面高度          │
│     (整个管体区域内搜索)         │
│  ④ 油面 vs 红线 → 状态判定      │
└─────────┬───────────────────────┘
          │
          ▼
    ┌──────────────────────────┐
    │ 油面在两线之间 → NORMAL   │
    │ 油面在两线之外 → ABNORMAL │
    │ 无法识别       → UNKNOWN  │
    └──────────────────────────┘
```

## 项目结构

```
油位表识别/
├── oil_level_gauge_algorithm_annotated.py   # 核心算法模块
├── api_minio_annotated.py                   # FastAPI + MinIO 服务
├── detect.py                                # 本地推理脚本
├── train.py                                 # YOLOv8 训练脚本
├── weights/
│   └── best.pt                              # 训练好的模型权重
├── dataset/
│   ├── data.yaml                            # 数据集配置
│   ├── train/
│   │   ├── images/                          # 训练集图片 (229 张)
│   │   └── labels/                          # 训练集标签 (YOLO 格式)
│   └── val/
│       ├── images/                          # 验证集图片 (26 张)
│       └── labels/                          # 验证集标签
└── results/                                 # 检测结果示例
```

## 环境依赖

```bash
pip install ultralytics opencv-python numpy fastapi uvicorn minio pydantic
```

## 快速开始

### 1. 本地推理

```bash
# 检测单张图片
python detect.py --image dataset/val/images/3809_2026_07_22_16_22_26_865.jpg

# 批量检测整个文件夹
python detect.py --image dataset/val/images/

# 指定模型权重路径
python detect.py --image dataset/val/images/ --model weights/best.pt

# 只看终端输出，不保存图片不弹窗
python detect.py --image dataset/val/images/ --no-save --no-show
```

### 2. 启动 API 服务

```bash
python api_minio_annotated.py
# 或
uvicorn api_minio_annotated:app --host 0.0.0.0 --port 8700
```

接口：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/detect/minio` | 批量油位检测 |

请求示例：

```json
{
  "object_keys": ["image001.jpg", "image002.jpg"],
  "params": {"det_conf": 0.3}
}
```

### 3. 模型训练

```bash
# 使用默认参数训练
python train.py

# 指定参数
python train.py --epochs 200 --batch 8 --model yolov8m.pt

# 训练完成后导出 ONNX
python train.py --export

# 从中断处恢复
python train.py --resume
```

训练好的权重保存在 `runs/train/oil_level_gauge/weights/best.pt`。

## 算法参数

所有参数均可通过 API 请求的 `params` 字段动态覆盖。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `det_conf` | 0.25 | YOLO 检测置信度阈值 |
| `det_iou` | 0.45 | NMS IoU 阈值 |
| `tube_expected_width_ratio` | 0.22 | 管体预期宽度比例 |
| `red_side_band_ratio` | 0.18 | 管体左侧红线容差带 |
| `level_min_score` | 2.0 | 油面梯度最低得分 |

## 返回字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `state` | string | `normal` / `abnormal` / `unknown` |
| `check_result` | string | `NORMAL` / `ABNORMAL` / `UNKNOWN` |
| `is_normal` | bool | 是否正常 |
| `reason` | string | 判定原因 |
| `confidence` | float | YOLO 检测置信度 |
| `gauge_bbox` | list | 油位表检测框 [x, y, w, h] |
| `tube_bounds` | list | 管体边界 [x1, y1, x2, y2] |
| `reference_lines_y` | list | 红线 y 坐标 |
| `oil_level_y` | int | 油面 y 坐标 |
| `oil_level_position_ratio` | float | 油面在红线区间内的相对位置 |
| `oil_level_score` | float | 油面检测梯度得分 |
| `result_image` | image | 标注后的可视化结果图 |
