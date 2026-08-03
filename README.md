# 油位表识别系统

基于 YOLO + 传统视觉的两段式油位表油位识别系统。自动检测图片中的油位表，定位液柱管体、红色参考线和油面高度，判定油位状态（NORMAL / ABNORMAL / UNKNOWN）。

## 算法流程

```
输入图片
  │
  ▼
┌───────────────────────────┐
│ 第一段：YOLO 目标检测      │  ← level_best.pt
│ 定位油位表区域 (bbox)      │
└─────────┬─────────────────┘
          │ 裁剪 ROI
          ▼
┌─────────────────────────────────────────┐
│ 第二段：结构化视觉分析                    │
│                                         │
│  ① Sobel 梯度 → 管体左右边界            │
│  ② 红色参考线识别（双策略）              │
│     优先：YOLO 红线模型 (red_best.pt)    │
│     回退：传统 HSV/Lab 颜色分割          │
│  ③ 亮度梯度 → 油面高度                   │
│  ④ 油面 vs 红线 → 状态判定               │
└─────────┬───────────────────────────────┘
          │
          ▼
    ┌──────────────────────────┐
    │ 油面在两线之间 → NORMAL   │
    │ 油面在两线之外 → ABNORMAL │
    │ 无法识别       → UNKNOWN  │
    └──────────────────────────┘
```

### 红线检测双策略

```
红线模型已加载？
├── 是 → YOLO 检测红线
│       ├── >= 2 条 → 使用（redline_method = "yolo"）
│       └── < 2 条  → 回退到传统方法
└── 否 → 直接传统方法

传统方法内部：
曝光判断 → 正常光照 / 严重曝光（Gamma + CLAHE 增强）
偏色判断 → 正常 / 偏蓝 / 偏黄（定向白平衡补偿）
普通 HSV 检测 → candidates < 2 → 增强算法 fallback
候选排序 → 多特征评分（面积 + 宽高比 + 水平梯度 + 连续性）
```

## 项目结构

```
油位表识别/
├── oil_level_gauge_algorithm_annotated.py   # 核心算法模块（传统视觉 + YOLO 融合）
├── oil_level_gauge_algorithm_yolo.py        # 纯 YOLO 红线检测版本
├── api_minio_annotated.py                   # FastAPI + MinIO 服务接口
├── test.py                                  # 本地推理测试脚本
├── train.py                                 # YOLOv8 训练脚本
├── level_best.pt                            # 油位表检测模型权重
├── red_best.pt                              # 红线检测模型权重
└── yolov8s.pt                               # YOLOv8 预训练权重
```

## 环境依赖

```bash
pip install ultralytics opencv-python numpy fastapi uvicorn minio pydantic
```

## 快速开始

### 1. 本地推理（推荐）

```bash
# 单张图片（双模型）
python test.py --image test.jpg

# 整个文件夹
python test.py --image ./test_images/

# 指定模型路径
python test.py --image test.jpg --gauge-model level_best.pt --redline-model red_best.pt

# 调整置信度阈值
python test.py --image test.jpg --gauge-conf 0.3 --redline-conf 0.25

# 指定输出目录
python test.py --image test.jpg --output ./results/
```

### 2. 启动 API 服务

```bash
python api_minio_annotated.py
# 或
uvicorn api_minio_annotated:app --host 0.0.0.0 --port 8700
```

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查（返回模型加载状态） |
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

## 算法特性

### 自适应光照环境

| 场景 | 检测策略 |
|------|----------|
| 正常光照 | 标准 HSV 红色分割（低开销） |
| 严重曝光 | Gamma 补偿 + CLAHE 增强 + HSV/RGB/Lab 融合检测 |
| 偏蓝冷光 | 定向白平衡 + R-B 显著性约束 + 紫红 Hue 扩展 |
| 偏黄暖光 | 定向白平衡 + Lab a*/b* 双通道约束 + R-G 显著性 + 暖色 Hue 扩展 |

### 红线候选多特征评分

候选红线的排序基于综合评分，而非单一颜色匹配：

- **面积评分**：候选区域像素面积
- **宽高比评分**：水平长条结构特征
- **水平梯度评分**：Sobel dy=1 检测红线上下边缘强度
- **连续性评分**：基于颜色 mask 的横向前景像素比例

### 自动 fallback 机制

- YOLO 红线检测不足 → 自动回退到传统 HSV/Lab 方法
- 正常光照 HSV 检测不足 → 自动回退到增强算法（Gamma + CLAHE + 偏色补偿）

## 算法参数

所有参数均可通过 API 请求的 `params` 字段或 `infer()` 的 `params` 参数动态覆盖。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `det_conf` | 0.25 | 油位表 YOLO 检测置信度阈值 |
| `det_iou` | 0.45 | 油位表检测 NMS IoU 阈值 |
| `redline_det_conf` | 0.25 | 红线 YOLO 检测置信度阈值 |
| `redline_det_iou` | 0.45 | 红线检测 NMS IoU 阈值 |
| `tube_expected_width_ratio` | 0.22 | 管体预期宽度比例 |
| `red_side_band_ratio` | 0.25 | 管体左侧红线搜索容差带比例 |
| `red_saturation_min` | 25 | 增强分支最小饱和度 |
| `red_value_min` | 40 | 增强分支最小明度 |
| `overexposure_threshold` | 0.22 | 曝光过度高亮像素占比阈值 |
| `red_gamma` | 0.7 | 曝光补偿 Gamma 值 |
| `level_min_score` | 1.5 | 油面梯度最低得分阈值 |

## 返回字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `state` | string | `normal` / `abnormal` / `unknown` |
| `check_result` | string | `NORMAL` / `ABNORMAL` / `UNKNOWN` |
| `is_normal` | bool | 是否正常 |
| `reason` | string | 判定原因标识 |
| `confidence` | float | 油位表 YOLO 检测置信度 |
| `gauge_bbox` | list | 油位表检测框 [x, y, w, h] |
| `tube_bounds` | list | 管体边界 [x1, y1, x2, y2] |
| `reference_lines_y` | list | 红线 y 坐标 [上, 下] |
| `oil_level_y` | int | 油面 y 坐标 |
| `oil_level_position_ratio` | float | 油面在管体内的相对位置 (0~1) |
| `oil_level_score` | float | 油面检测梯度得分 |
| `is_overexposed` | bool | 是否严重曝光过度 |
| `color_tint` | string | 偏色方向：`normal` / `blue` / `yellow` |
| `redline_method` | string | 红线检测方法：`yolo` / `traditional` |
| `result_image` | image | 标注后的可视化结果图 |
| `cost_ms` | int | 推理耗时（毫秒） |
