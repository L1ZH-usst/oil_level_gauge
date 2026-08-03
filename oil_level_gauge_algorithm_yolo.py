"""
油位表油位识别算法 - 核心模块

本模块实现了"两段式"油位表识别流程：
  第一段：使用 YOLO 模型检测图片中的油位表目标区域（bounding box）
  第二段：在检测框内，使用传统视觉方法定位液柱管体、红色参考线、油面高度，
          最终判定油位状态（NORMAL / ABNORMAL / UNKNOWN）

判定规则：
  - 油面 y 坐标在上下两条红线之间 → NORMAL（正常）
  - 油面 y 坐标在红线范围之外     → ABNORMAL（异常）
  - 红线或油面证据不足时          → UNKNOWN（无法判定）

依赖：
  - ultralytics (YOLO) — 目标检测
  - opencv-python (cv2) — 图像处理
  - numpy — 数值计算
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


class OilLevelGaugeDetector:
    """
    油位表检测器主类。

    职责：
      1. 加载 YOLO 模型权重
      2. 对外暴露 infer() 方法，接收一张图片，返回完整的油位识别结果

    典型用法：
        detector = OilLevelGaugeDetector("weights/best.pt")
        detector.load()
        result = detector.infer(image_bgr, params={"det_conf": 0.3})
    """

    def __init__(self, model_path: str | Path, defaults: dict[str, Any] | None = None):
        """
        初始化检测器（不加载模型，需要显式调用 load()）。

        参数:
            model_path: YOLO 模型权重文件路径（.pt 文件）
            defaults:   默认算法参数字典，可被 infer() 的 params 覆盖
        """
        self.model_path = str(model_path)
        self.defaults = defaults or {}
        self._model: YOLO | None = None       # 油位表 YOLO 模型实例（延迟加载）
        self._loaded = False                   # 油位表模型是否已加载
        self._redline_model: YOLO | None = None  # 红线检测 YOLO 模型实例（延迟加载）
        self._redline_loaded = False           # 红线模型是否已加载

    @property
    def loaded(self) -> bool:
        """返回模型是否已成功加载。"""
        return self._loaded

    def load(self) -> None:
        """
        加载 YOLO 模型权重到内存。

        如果权重文件不存在会抛出 RuntimeError。
        """
        if not Path(self.model_path).exists():
            raise RuntimeError(f"油位表模型不存在: {self.model_path}")
        self._model = YOLO(self.model_path)
        self._loaded = True

    def load_redline_model(self, redline_model_path: str | Path) -> None:
        """
        加载红线检测 YOLO 模型权重到内存（可选，不调用则使用传统方法检测红线）。

        如果权重文件不存在会抛出 RuntimeError。
        """
        redline_model_path = str(redline_model_path)
        if not Path(redline_model_path).exists():
            raise RuntimeError(f"红线检测模型不存在: {redline_model_path}")
        self._redline_model = YOLO(redline_model_path)
        self._redline_loaded = True

    def infer(self, image: Any, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        对单张图片执行完整的油位识别流水线。

        流水线步骤：
          1. YOLO 目标检测 → 定位油位表区域
          2. 裁剪检测框 → 得到 ROI（感兴趣区域）
          3. 液柱管体定位 → 确定管体左右边界
          4. 红色参考线识别 → 在管体左边界左侧区域找到上下两条红线
          5. 油面高度定位 → 在管体中心区域搜索油面
          6. 状态判定 → 油面是否在红线之间

        参数:
            image:  BGR 格式的 numpy 数组（OpenCV 读入的图片）
            params: 可选的算法参数覆盖字典，会与 defaults 合并

        返回:
            包含以下关键字段的字典：
              - cost_ms:          推理耗时（毫秒）
              - state:            状态 "normal" / "abnormal" / "unknown"
              - check_result:     大写状态 "NORMAL" / "ABNORMAL" / "UNKNOWN"
              - is_normal:        是否正常（布尔值）
              - reason:           判定原因描述
              - confidence:       YOLO 检测置信度
              - gauge_bbox:       油位表检测框 [x, y, w, h]
              - tube_bounds:      液柱管体边界 [x1, y1, x2, y2]（绝对坐标）
              - reference_lines_y: 两条红线的 y 坐标列表（绝对坐标）
              - oil_level_y:      油面 y 坐标（绝对坐标）
              - oil_level_position_ratio: 油面在液柱管体内的相对位置 (0~1)
              - oil_level_score:  油面检测梯度得分
              - result_image:     标注了检测结果的可视化图片
              - image_width/height: 原图尺寸
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("油位表模型未加载")

        t0 = time.time()

        # 合并默认参数和运行时参数（运行时参数优先）
        options = dict(self.defaults)
        options.update(params or {})

        # === 第一段：YOLO 目标检测 ===
        gauge = _detect_gauge(self._model, image, options)
        if gauge is None:
            # 没有检测到油位表目标，返回 UNKNOWN
            return _unknown_result(
                image=image,
                reason="oil_level_gauge_not_detected",
                cost_ms=int((time.time() - t0) * 1000),
                roi=image,  # 没有 ROI，用原图做曝光判断
            )

        # 裁剪检测框区域，得到 ROI（感兴趣区域）
        # bbox_xyxy: [x1, y1, x2, y2] 格式
        # bbox_xywh: [x, y, w, h] 格式
        roi, bbox_xyxy, bbox_xywh = _crop_bbox(image, gauge["bbox_xyxy"])

        # === 第二段第一步：定位液柱管体左右边界 ===
        tube = _find_tube_bounds(roi, options)
        if tube is None:
            # 管体定位失败
            return _unknown_result(
                image=_draw_detection_only(image, bbox_xyxy, gauge["confidence"]),
                reason="tube_region_not_found",
                cost_ms=int((time.time() - t0) * 1000),
                confidence=gauge["confidence"],
                gauge_bbox=bbox_xywh,
                roi=roi,
            )

        # === 第二段第二步：识别两条红色参考线（YOLO 优先，传统方法 fallback）===
        redline_method = "traditional"

        if self._redline_loaded and self._redline_model is not None:
            # 优先使用 YOLO 红线检测模型
            lines = _detect_redlines_with_yolo(self._redline_model, roi, options)
            if len(lines) >= 2:
                redline_method = "yolo"
            else:
                # YOLO 检测不足，fallback 到传统方法
                lines = _find_reference_lines(roi, tube, options)
        else:
            # 未加载红线模型，使用传统方法
            lines = _find_reference_lines(roi, tube, options)

        if len(lines) < 2:
            # 找不到足够的红色参考线（至少需要上下两条）
            return _unknown_result(
                image=_draw_structure_result(image, bbox_xyxy, gauge["confidence"], tube, [], None, "unknown"),
                reason="reference_red_lines_not_found",
                cost_ms=int((time.time() - t0) * 1000),
                confidence=gauge["confidence"],
                gauge_bbox=bbox_xywh,
                tube_bounds=_abs_tube_bounds(tube, bbox_xyxy),
                roi=roi,
            )

        # lines[0] 是上方红线（y 较小），lines[1] 是下方红线（y 较大）
        upper_line, lower_line = lines[0], lines[1]

        # === 第二段第三步：在整个管体区域内定位油面高度 ===
        level = _find_oil_level(roi, tube, options)
        if level is None:
            # 油面定位失败（梯度不够强，证据不足）
            return _unknown_result(
                image=_draw_structure_result(image, bbox_xyxy, gauge["confidence"], tube, lines, None, "unknown"),
                reason="oil_level_not_found",
                cost_ms=int((time.time() - t0) * 1000),
                confidence=gauge["confidence"],
                gauge_bbox=bbox_xywh,
                tube_bounds=_abs_tube_bounds(tube, bbox_xyxy),
                reference_lines_y=_abs_line_positions(lines, bbox_xyxy),
                roi=roi,
            )

        # === 状态判定 ===
        upper_y = int(upper_line["center_y"])   # 上红线中心 y（ROI 内坐标）
        lower_y = int(lower_line["center_y"])   # 下红线中心 y（ROI 内坐标）
        oil_y = int(level["y"])                  # 油面 y 坐标（ROI 内坐标）

        # 核心判定：油面 y 在上下红线之间即为正常
        is_normal = upper_y <= oil_y <= lower_y
        state = "normal" if is_normal else "abnormal"
        check_result = "NORMAL" if is_normal else "ABNORMAL"
        reason = "oil_level_between_red_lines" if is_normal else "oil_level_outside_red_lines"

        # 新：基于管体顶部/底部计算
        tube_top = tube["top"]
        tube_bottom = tube["bottom"]
        position_ratio = (oil_y - tube_top) / max(tube_bottom - tube_top, 1)

        # 生成带标注的可视化结果图
        result_image = _draw_structure_result(image, bbox_xyxy, gauge["confidence"], tube, lines, level, state)

        # 组装返回结果（detections 数组是为了兼容多目标扩展，当前始终只有 1 个）
        return {
            "cost_ms": int((time.time() - t0) * 1000),
            "detections": [
                {
                    "class": "oil_level_gauge",
                    "state": state,
                    "check_result": check_result,
                    "is_normal": is_normal,
                    "reason": reason,
                    "confidence": round(gauge["confidence"], 4),
                    "gauge_bbox": bbox_xywh,
                    "tube_bounds": _abs_tube_bounds(tube, bbox_xyxy),
                    "reference_lines_y": _abs_line_positions(lines, bbox_xyxy),
                    "oil_level_y": int(bbox_xyxy[1] + oil_y),         # 转为原图绝对坐标
                    "oil_level_position_ratio": round(float(position_ratio), 4),
                    "oil_level_score": round(float(level["score"]), 4),
                }
            ],
            "state": state,
            "check_result": check_result,
            "is_normal": is_normal,
            "reason": reason,
            "confidence": round(gauge["confidence"], 4),
            "gauge_bbox": bbox_xywh,
            "tube_bounds": _abs_tube_bounds(tube, bbox_xyxy),
            "reference_lines_y": _abs_line_positions(lines, bbox_xyxy),
            "oil_level_y": int(bbox_xyxy[1] + oil_y),
            "oil_level_position_ratio": round(float(position_ratio), 4),
            "oil_level_score": round(float(level["score"]), 4),
            "result_image": result_image,
            "image_width": image.shape[1],
            "image_height": image.shape[0],
            "is_overexposed": _is_overexposed(
                roi,
                threshold=float(options.get("overexposure_threshold", 0.22)),
                highlight_val=int(options.get("overexposure_highlight_val", 220)),
            ),
            "color_tint": _diagnose_color_tint(roi),
            "redline_method": redline_method,
        }


# ===========================================================================
#  辅助函数：结果构建
# ===========================================================================

def _unknown_result(
    image: np.ndarray,
    reason: str,
    cost_ms: int,
    confidence: float = 0.0,
    gauge_bbox: list[int] | None = None,
    tube_bounds: list[int] | None = None,
    reference_lines_y: list[int] | None = None,
    roi: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    构建 UNKNOWN 状态的返回结果。

    当流水线中任何一步失败（未检测到油位表、管体定位失败、红线不足、油面找不到）
    都调用此函数统一返回。这是一种保守策略：宁可返回 UNKNOWN 也不误判为异常。

    参数:
        image:             原图或已标注的中间结果图
        reason:            失败原因标识（如 "oil_level_gauge_not_detected"）
        cost_ms:           到目前为止的耗时
        confidence:        YOLO 检测置信度（如果有）
        gauge_bbox:        油位表检测框（如果有）
        tube_bounds:       管体边界（如果有）
        reference_lines_y: 红线位置（如果有）
        roi:               检测框裁剪区域（如果有），用于曝光判断；没有时用 image
    """
    # 用 ROI 判断曝光（如果有的话），否则回退到用原图
    overexposed_img = roi if roi is not None else image
    is_overexposed = _is_overexposed(overexposed_img, 0.22, 220)

    return {
        "cost_ms": cost_ms,
        "detections": [
            {
                "class": "oil_level_gauge",
                "state": "unknown",
                "check_result": "UNKNOWN",
                "is_normal": False,
                "reason": reason,
                "confidence": round(float(confidence), 4),
                "gauge_bbox": gauge_bbox or [],
                "tube_bounds": tube_bounds or [],
                "reference_lines_y": reference_lines_y or [],
                "oil_level_y": None,
                "oil_level_position_ratio": None,
                "oil_level_score": 0.0,
            }
        ],
        "state": "unknown",
        "check_result": "UNKNOWN",
        "is_normal": False,
        "reason": reason,
        "confidence": round(float(confidence), 4),
        "gauge_bbox": gauge_bbox or [],
        "tube_bounds": tube_bounds or [],
        "reference_lines_y": reference_lines_y or [],
        "oil_level_y": None,
        "oil_level_position_ratio": None,
        "oil_level_score": 0.0,
        "result_image": _draw_unknown_banner(image, reason),
        "image_width": image.shape[1],
        "image_height": image.shape[0],
        "is_overexposed": is_overexposed,
        "color_tint": _diagnose_color_tint(overexposed_img),
    }


# ===========================================================================
#  第一段：YOLO 目标检测
# ===========================================================================

def _detect_gauge(model: YOLO, image: np.ndarray, options: dict[str, Any]) -> dict[str, Any] | None:
    """
    使用 YOLO 模型检测图片中的油位表目标。

    工作原理：
      - 将原图输入 YOLO 模型，获取所有检测结果
      - 在所有检测框中，选择置信度最高的一个作为最终结果
      - 如果没有任何检测结果，返回 None

    参数:
        model:   已加载的 YOLO 模型实例
        image:   BGR 格式的输入图片
        options: 算法参数字典
            - det_conf: 检测置信度阈值（默认 0.25），低于此值的检测结果被过滤
            - det_iou:  NMS 的 IoU 阈值（默认 0.45），重叠度高于此值的框被合并

    返回:
        成功时返回 {"bbox_xyxy": [x1,y1,x2,y2], "confidence": float}
        失败时返回 None
    """
    conf = float(options.get("det_conf", 0.25))  # 置信度阈值
    iou = float(options.get("det_iou", 0.45))    # NMS IoU 阈值
    results = model(image, conf=conf, iou=iou, verbose=False)
    if not results:
        return None

    # 获取第一个结果的所有检测框
    boxes = getattr(results[0], "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None

    # 选择置信度最高的检测框（当前模型只检测油位表一类，通常只有 0~1 个框）
    best = max(boxes, key=lambda box: float(box.conf[0]))

    # xyxy 格式: [左上角x, 左上角y, 右下角x, 右下角y]
    x1, y1, x2, y2 = best.xyxy[0].tolist()
    return {
        "bbox_xyxy": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
        "confidence": float(best.conf[0]),
    }


# ===========================================================================
#  ROI 裁剪
# ===========================================================================

def _crop_bbox(image: np.ndarray, bbox_xyxy: list[int]) -> tuple[np.ndarray, list[int], list[int]]:
    """
    从原图中裁剪出检测框区域（ROI），并做边界安全检查。

    参数:
        image:     原图
        bbox_xyxy: 检测框坐标 [x1, y1, x2, y2]

    返回:
        (roi, bbox_xyxy, bbox_xywh)
        - roi:       裁剪后的小图（numpy 数组的副本）
        - bbox_xyxy: 边界修正后的 [x1, y1, x2, y2]
        - bbox_xywh: [x, y, width, height] 格式（用于外部接口返回）
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    # 将坐标限制在图片范围内，防止越界
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))  # 确保 x2 > x1，裁剪区域不为空
    y2 = max(y1 + 1, min(h, y2))
    return image[y1:y2, x1:x2].copy(), [x1, y1, x2, y2], [x1, y1, x2 - x1, y2 - y1]


# ===========================================================================
#  第二段步骤一：液柱管体定位
# ===========================================================================

def _find_tube_bounds(roi: np.ndarray, options: dict[str, Any]) -> dict[str, int] | None:
    """
    在油位表 ROI 中定位液柱管体的左右边界。

    算法原理：
      油位表的核心是一个竖直的透明/半透明液柱管，管壁是两条明显的竖直边缘。
      通过 Sobel 算子计算水平方向梯度，管壁处会产生强烈的梯度峰值。
      算法在梯度图中寻找最优的一对峰值（左边缘 + 右边缘），即为管体左右边界。

    步骤：
      1. 灰度化 → Sobel 水平梯度 → 按列求平均 → 得到一维梯度剖面
      2. 平滑梯度曲线（消除噪声）
      3. 找出梯度曲线中的所有局部极大值点
      4. 在所有峰值对中，选出综合评分最高的一对作为管体边界

    评分标准：
      - 两个边缘的梯度强度之和（越高越好）
      - 管体中心是否接近图片中心（越居中越好）
      - 管体宽度是否接近预期值（越接近越好）

    参数:
        roi:     油位表区域的小图
        options: 算法参数
            - tube_smooth_kernel:       平滑核大小（默认自适应）
            - tube_search_left_ratio:   搜索范围左边界比例（默认 0.15）
            - tube_search_right_ratio:  搜索范围右边界比例（默认 0.85）
            - tube_min_width_ratio:     管体最小宽度比例（默认 0.08）
            - tube_max_width_ratio:     管体最大宽度比例（默认 0.45）
            - tube_expected_width_ratio:预期宽度比例（默认 0.22）

    返回:
        成功时返回 {"left": x, "right": x, "top": 0, "bottom": h-1}
        失败时返回 None
    """
    # 步骤 1：计算水平方向梯度
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Sobel 算子计算 x 方向梯度，取绝对值后按列求平均，得到一维梯度曲线
    # 梯度高的位置对应竖直边缘（即管壁）
    grad_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)).mean(axis=0)
    if grad_x.size < 20:
        return None  # 图片太窄，无法分析

    # 步骤 2：平滑梯度曲线，消除毛刺噪声
    kernel_size = int(options.get("tube_smooth_kernel", max(9, (roi.shape[1] // 25) | 1)))
    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    smooth = np.convolve(grad_x, kernel, mode="same")

    # 步骤 3：在搜索范围内找出所有局部极大值点
    # 不搜索边缘区域，避免边界干扰
    left_limit = max(1, int(roi.shape[1] * float(options.get("tube_search_left_ratio", 0.15))))
    right_limit = min(roi.shape[1] - 2, int(roi.shape[1] * float(options.get("tube_search_right_ratio", 0.85))))

    peaks = []
    for idx in range(left_limit, right_limit):
        # 局部极大值：比左右邻居都大（或相等）
        if smooth[idx] >= smooth[idx - 1] and smooth[idx] >= smooth[idx + 1]:
            peaks.append(idx)
    # 只保留梯度最强的 40 个峰值，减少后续组合搜索量
    peaks = sorted(peaks, key=lambda idx: smooth[idx], reverse=True)[:40]
    if len(peaks) < 2:
        return None  # 峰值不足，无法组成一对边缘

    # 步骤 4：在所有峰值对中搜索最优的管体边界组合
    min_width = roi.shape[1] * float(options.get("tube_min_width_ratio", 0.08))
    max_width = roi.shape[1] * float(options.get("tube_max_width_ratio", 0.45))
    expected_width = roi.shape[1] * float(options.get("tube_expected_width_ratio", 0.22))
    image_center = roi.shape[1] / 2.0

    best: tuple[float, int, int] | None = None
    for left in peaks:
        for right in peaks:
            if right <= left:
                continue
            width = right - left
            if width < min_width or width > max_width:
                continue  # 宽度不在合理范围内
            center = (left + right) / 2.0
            # 综合评分 = 边缘强度 - 偏离中心的惩罚 - 宽度偏离预期的惩罚
            score = float(smooth[left] + smooth[right])
            score -= abs(center - image_center) * 0.08  # 偏离中心越远，扣分越多
            score -= abs(width - expected_width) * 0.04  # 宽度越偏离预期，扣分越多
            if best is None or score > best[0]:
                best = (score, left, right)

    if best is None:
        return None

    left, right = int(best[1]), int(best[2])
    if right - left < 20:
        return None  # 管体太窄，不合理

    # 返回管体边界（top/bottom 与 ROI 等高，实际只用 left/right）
    return {
        "left": left,
        "right": right,
        "top": 0,
        "bottom": roi.shape[0] - 1,
    }


# ===========================================================================
#  第二段步骤二：红色参考线识别（仅管体左边界左侧区域）
# ===========================================================================

def _is_overexposed(roi: np.ndarray, threshold: float = 0.22, highlight_val: int = 220) -> bool:
    """
    判断 ROI 区域是否属于严重曝光过度。

    原理：
        1. 计算灰度图中高亮像素（亮度 >= 220）在整张图中的占比。
        2. 如果高亮像素比例超过设定阈值（默认 22%），则判定为严重曝光。

    参数:
        roi: 输入的 BGR 图片
        threshold: 高亮像素占比阈值，默认 0.22 (22%)
        highlight_val: 认定为高亮的像素亮度门槛，默认 220
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    highlight_pixels = np.sum(gray >= highlight_val)
    total_pixels = gray.size
    ratio = highlight_pixels / total_pixels
    return ratio > threshold


def _diagnose_color_tint(roi: np.ndarray) -> str:
    """
    判断图像的偏色方向。

    返回:
        "blue": 偏蓝/冷光
        "yellow": 偏黄/暖光
        "normal": 色温基本正常
    """
    roi_float = roi.astype(np.float32)
    avg_b = np.mean(roi_float[:, :, 0])
    avg_g = np.mean(roi_float[:, :, 1])
    avg_r = np.mean(roi_float[:, :, 2])

    # 方式：通过 R 与 B 的相对比例判断色温偏移
    b_r_ratio = avg_b / (avg_r + 1e-5)
    r_b_ratio = avg_r / (avg_b + 1e-5)

    if b_r_ratio > 1.25:  # 蓝色显著高于红色
        return "blue"
    elif r_b_ratio > 1.03:  # 红色/黄色显著高于蓝色
        return "yellow"
    else:
        return "normal"


def _find_reference_lines_enhanced(
        roi: np.ndarray,
        options: dict[str, Any],
) -> np.ndarray:
    """
    增强版红线检测掩膜生成（带偏色自动诊断与定向补偿机制）
    """
    # ==========================================================
    # 1. 偏色方向诊断 (Blue / Yellow / Normal)
    # ==========================================================
    tint_type = _diagnose_color_tint(roi)

    # ==========================================================
    # 2. 根据偏色类型应用“定向”白平衡与颜色补偿
    # ==========================================================
    roi_float = roi.astype(np.float32)
    avg_b = np.mean(roi_float[:, :, 0])
    avg_g = np.mean(roi_float[:, :, 1])
    avg_r = np.mean(roi_float[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3.0 + 1e-5

    roi_wb = roi_float.copy()

    if tint_type == "blue":
        # 常规全局白平衡
        roi_wb[:, :, 0] = np.clip(roi_float[:, :, 0] * (avg_gray / (avg_b + 1e-5)), 0, 255)
        roi_wb[:, :, 1] = np.clip(roi_float[:, :, 1] * (avg_gray / (avg_g + 1e-5)), 0, 255)
        roi_wb[:, :, 2] = np.clip(roi_float[:, :, 2] * (avg_gray / (avg_r + 1e-5)), 0, 255)

    elif tint_type == "yellow":
        # 【暖色/偏黄定向校正】：重点拉高 B 通道，压低过度饱和的 R/G 通道
        roi_wb[:, :, 0] = np.clip(roi_float[:, :, 0] * (avg_gray / (avg_b + 1e-5)) * 1.20, 0, 255)
        roi_wb[:, :, 1] = np.clip(roi_float[:, :, 1] * (avg_gray / (avg_g + 1e-5)) * 0.90, 0, 255)
        roi_wb[:, :, 2] = np.clip(roi_float[:, :, 2] * (avg_gray / (avg_r + 1e-5)) * 0.90, 0, 255)

    else:
        # 常规全局白平衡
        roi_wb[:, :, 0] = np.clip(roi_float[:, :, 0] * (avg_gray / (avg_b + 1e-5)), 0, 255)
        roi_wb[:, :, 1] = np.clip(roi_float[:, :, 1] * (avg_gray / (avg_g + 1e-5)), 0, 255)
        roi_wb[:, :, 2] = np.clip(roi_float[:, :, 2] * (avg_gray / (avg_r + 1e-5)), 0, 255)

    roi_balanced = roi_wb.astype(np.uint8)

    # ==========================================================
    # 3. Gamma 曝光补偿 + CLAHE 增强
    # ==========================================================
    gamma = float(options.get("red_gamma", 0.7))
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(256)]).astype("uint8")
    roi_gamma = cv2.LUT(roi_balanced, table)

    lab = cv2.cvtColor(roi_gamma, cv2.COLOR_BGR2LAB)
    l, a, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b_channel]), cv2.COLOR_LAB2BGR)

    # ==========================================================
    # 4. 定向 HSV 与 颜色显著性掩膜提取
    # ==========================================================
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    b_ch, g_ch, r_ch = cv2.split(enhanced)

    sat_min = int(options.get("red_saturation_min", 25))
    val_min = int(options.get("red_value_min", 40))

    # 基础红色区间 (H: 0~12 和 155~179)
    mask_hsv_red = cv2.inRange(hsv, np.array([0, sat_min, val_min]), np.array([15, 255, 255]))
    mask_hsv_red2 = cv2.inRange(hsv, np.array([155, sat_min, val_min]), np.array([179, 255, 255]))
    hsv_mask = cv2.bitwise_or(mask_hsv_red, mask_hsv_red2)

    if tint_type == "yellow":
        # ------------------------------------------------------
        # 分支 A：偏黄场景专属补偿机制
        # ------------------------------------------------------
        # 1. 扩展橙黄色 Hue 区间 (H: 13~28)
        mask_orange_yellow = cv2.inRange(
            hsv, np.array([13, min(255, sat_min + 15), val_min]), np.array([50, 255, 255])
        )
        hsv_mask = cv2.bitwise_or(hsv_mask, mask_orange_yellow)

        # 2. 强效 R-G 显著性算子 (变黄的红线 R 依然显著高于 G，但黄色背景 R 接近 G)
        rg_diff = r_ch.astype(np.int16) - g_ch.astype(np.int16)
        salience_mask = np.zeros_like(r_ch)
        salience_mask[rg_diff > 10] = 255

    elif tint_type == "blue":
        # ------------------------------------------------------
        # 分支 B：偏蓝场景专属补偿机制
        # ------------------------------------------------------
        # 1. 扩展紫红/粉红 Hue 区间 (H: 140~154)
        mask_purple_red = cv2.inRange(
            hsv, np.array([140, sat_min, val_min]), np.array([154, 255, 255])
        )
        hsv_mask = cv2.bitwise_or(hsv_mask, mask_purple_red)

        # 2. 强效 R-B 显著性算子 (偏蓝环境下，被压制的红线 R 依然要大于 B)
        rb_diff = r_ch.astype(np.int16) - b_ch.astype(np.int16)
        salience_mask = np.zeros_like(r_ch)
        salience_mask[rb_diff > 15] = 255

    else:
        # ------------------------------------------------------
        # 分支 C：常规场景
        # ------------------------------------------------------
        rg_diff = r_ch.astype(np.int16) - (g_ch.astype(np.int16) + b_ch.astype(np.int16)) / 2
        salience_mask = np.zeros_like(r_ch)
        salience_mask[rg_diff > 20] = 255

    # CIE Lab 空间中的 a* 通道 (> 134 偏红/偏橙，抗光照能力强)
    lab_salience = np.zeros_like(a)
    lab_salience[a > 134] = 255

    # 合并显著性区域
    color_salience = cv2.bitwise_or(salience_mask, lab_salience)

    # 掩膜求交集（HSV 空间 ∩ 显著性空间，确保不误杀且不误吸）
    mask = cv2.bitwise_and(hsv_mask, color_salience)

    # ==========================================================
    # 5. 形态学闭运算粘合
    # ==========================================================
    kernel = np.ones((5, 25), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    kernel2 = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel2)

    return mask


def _horizontal_gradient_score(
    roi_gray: np.ndarray,
    mask: np.ndarray,
    x: int, y: int, w: int, h: int,
) -> float:
    """
    新增：水平梯度结构评分 + 红线连续性评分

    工业油位表红色参考线具有明确的结构特征：
      - 上下边缘有明显的水平梯度（Sobel dy=1 响应强）
      - 在颜色 mask 中横向连续，中间断裂少

    本函数综合两个指标，返回 0~100 的结构评分。

    计算步骤：
      1. Sobel Y方向梯度：提取候选区域灰度 patch，计算 dy=1 梯度幅值均值
         → 反映水平边缘强度（红线上下边界）
      2. 红线连续性：使用已有颜色 mask（非灰度 Otsu），统计候选区域内
         每行 mask 前景像素占比，取平均值
         → 反映红线是否连续完整（避免白色反光/黄色污渍干扰）

    参数:
        roi_gray: ROI 的灰度图
        mask:     红线颜色检测的二值 mask（与连通域分析使用同一张）
        x, y, w, h: 候选框的 bbox（ROI 内坐标）
    返回:
        0~100 的浮点评分（梯度 0~50 + 连续性 0~50）
    """
    # 边界安全裁剪
    roi_h, roi_w = roi_gray.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(roi_w, x + w)
    y2 = min(roi_h, y + h)
    patch_gray = roi_gray[y1:y2, x1:x2]
    patch_mask = mask[y1:y2, x1:x2]
    if patch_gray.size == 0:
        return 0.0

    # --- 指标 1：水平梯度强度 ---
    # Sobel dy=1 检测水平边缘（红线上下边界）
    grad_y = np.abs(cv2.Sobel(patch_gray, cv2.CV_32F, 0, 1, ksize=3))
    # 均值归一化到 0~50
    gradient_score = min(float(np.mean(grad_y)) / 4.0, 50.0)

    # --- 指标 2：红线连续性（基于颜色 mask）---
    # 使用已有红线 mask 的前景像素比例，而非灰度 Otsu
    # 避免白色反光、黄色污渍、标签等非红色区域被误判为连续
    row_ratios = np.mean(patch_mask > 0, axis=1)
    max_row_ratio = np.max(row_ratios)
    continuity_score = (np.mean(row_ratios) * 30 + max_row_ratio * 20)

    return gradient_score + continuity_score


def _detect_redlines_with_yolo(
    model: YOLO,
    roi: np.ndarray,
    options: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    使用 YOLO 红线检测模型在 ROI 内检测红色参考线。

    工作原理：
      - 将 ROI 区域输入红线检测 YOLO 模型
      - 获取所有红线检测框，取每个检测框的 y 坐标中值作为红线位置
      - 按 y 中值升序排列，返回最上方和最下方的两条红线
      - 返回格式与 _find_reference_lines 一致（[{"center_y": float}, ...]）

    参数:
        model:   已加载的红线检测 YOLO 模型实例
        roi:     油位表区域的小图（BGR 格式）
        options: 算法参数字典
            - redline_det_conf: 红线检测置信度阈值（默认 0.25）
            - redline_det_iou:  红线检测 NMS IoU 阈值（默认 0.45）

    返回:
        按 y 中值升序排列的红线列表 [{"center_y": float}, ...]
        最多返回 2 条（最上和最下），不足 2 条时返回已有条目
    """
    conf = float(options.get("redline_det_conf", 0.25))
    iou = float(options.get("redline_det_iou", 0.45))

    results = model(roi, conf=conf, iou=iou, verbose=False)
    if not results:
        return []

    boxes = getattr(results[0], "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    # 提取所有检测框的 y 坐标中值
    detections = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        median_y = (y1 + y2) / 2.0  # 取 bbox 的 y 坐标中值
        detections.append({"center_y": float(median_y)})

    if len(detections) == 0:
        return []

    # 按 center_y 升序排列
    detections.sort(key=lambda d: d["center_y"])

    if len(detections) >= 2:
        # 取最上方（y 最小）和最下方（y 最大）的两条红线
        return [detections[0], detections[-1]]
    else:
        return detections


def _find_reference_lines(
    roi: np.ndarray,
    tube: dict[str, int],
    options: dict[str, Any]
) -> list[dict[str, Any]]:
    """
    工业油位表红色参考线检测（带曝光判断的双分支 + fallback 版本）

    逻辑：
        检测 ROI 的曝光程度：
        - 光照正常 (False) -> 运行原版算法（低开销 + 标准 HSV 色彩分割）
                              若 candidates < 2，自动 fallback 到增强算法
        - 曝光严重 (True)  -> 直接运行增强版算法（Gamma + CLAHE + RGB差值）
    """
    roi_h, roi_w = roi.shape[:2]

    # ==========================================================
    # 0. 曝光程度自动判定
    # ==========================================================
    is_overexposed = _is_overexposed(
        roi,
        threshold=float(options.get("overexposure_threshold", 0.22)),
        highlight_val=int(options.get("overexposure_highlight_val", 220))
    )

    # ----------------------------------------------------------
    # 分支一：光照正常（原版高效算法）
    # ----------------------------------------------------------
    if not is_overexposed:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 正常光照下的 HSV 容忍范围
        sat_min = int(options.get("red_saturation_min_normal", 80))
        val_min = int(options.get("red_value_min_normal", 80))

        mask_low = cv2.inRange(
            hsv,
            np.array([int(options.get("red_hue_low_min", 0)), sat_min, val_min]),
            np.array([int(options.get("red_hue_low_max", 12)), 255, 255]),
        )
        mask_high = cv2.inRange(
            hsv,
            np.array([int(options.get("red_hue_high_min", 160)), sat_min, val_min]),
            np.array([int(options.get("red_hue_high_max", 179)), 255, 255]),
        )
        mask = cv2.bitwise_or(mask_low, mask_high)

        # 原版精细形态学闭运算
        kernel = np.ones((3, 9), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # ----------------------------------------------------------
    # 分支二：严重曝光（增强版算法）
    # ----------------------------------------------------------
    else:
        # 新增：Gamma曝光补偿
        mask = _find_reference_lines_enhanced(roi, options)


    # ==========================================================
    # 公共步骤：掩膜区域限制（仅保留管体左侧）
    # ==========================================================
    tube_left = tube["left"]
    side_margin = int(roi_w * float(options.get("red_side_band_ratio", 0.25)))
    x_limit = min(roi_w, tube_left + side_margin)

    roi_mask = np.zeros_like(mask)
    roi_mask[:, :x_limit] = 255
    mask = cv2.bitwise_and(mask, roi_mask)


    # ==========================================================
    # 公共步骤：连通域分析 + 几何筛选 + 评分去重
    # ==========================================================
    # 新增：水平梯度评分需要灰度图（仅计算一次）
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    candidates = []

    min_area = int(roi_h * roi_w * float(options.get("red_min_area_ratio", 0.0003)))
    min_width = roi_w * float(options.get("red_min_width_ratio", 0.05))

    # 之前讨论过的垂直范围限制（如需限制在 15%~50% 范围可从 options 取参，默认 0~1.0 全高）
    min_y_ratio = float(options.get("red_min_y_ratio", 0.10))
    max_y_ratio = float(options.get("red_max_y_ratio", 0.50))

    for i in range(1, count):
        x, y, w, h, area = stats[i]

        if area < min_area or w < min_width:
            continue

        aspect = w / max(h, 1)
        if aspect < 2.0:  # 红线必须明显横向
            continue

        center_x = x + w / 2.0
        center_y = y + h / 2.0

        # Y 轴区间过滤
        if not (roi_h * min_y_ratio <= center_y <= roi_h * max_y_ratio):
            continue

        # 排除左贴边
        if x < 2:
            continue

        # 排除过度侵入管体内部的区域
        if x + w > tube_left + side_margin:
            continue

        # 修改：红线综合评分 = 面积 + 宽高比 + 水平梯度结构
        grad_score = _horizontal_gradient_score(roi_gray, mask, x, y, w, h)
        score = float(area) * 0.25 + float(aspect) * 20.0 + grad_score * 3.0

        candidates.append({
            "bbox": [int(x), int(y), int(w), int(h)],
            "center_x": float(center_x),
            "center_y": float(center_y),
            "area": int(area),
            "aspect": float(aspect),
            "score": score
        })

    # 新增：普通检测失败后的增强fallback
    # 如果当前走的是正常曝光分支，但候选不足，自动用增强算法重试
    if len(candidates) < 2 and not is_overexposed:
        mask_enhanced = _find_reference_lines_enhanced(roi, options)
        # 应用相同的区域限制
        roi_mask = np.zeros_like(mask_enhanced)
        roi_mask[:, :x_limit] = 255
        mask_enhanced = cv2.bitwise_and(mask_enhanced, roi_mask)

        count2, labels2, stats2, centroids2 = cv2.connectedComponentsWithStats(mask_enhanced, 8)
        candidates = []
        for i in range(1, count2):
            x, y, w, h, area = stats2[i]
            if area < min_area or w < min_width:
                continue
            aspect = w / max(h, 1)
            if aspect < 2.0:
                continue
            center_x = x + w / 2.0
            center_y = y + h / 2.0
            if not (roi_h * min_y_ratio <= center_y <= roi_h * max_y_ratio):
                continue
            if x < 2:
                continue
            if x + w > tube_left + side_margin:
                continue
            # 修改：红线综合评分 = 面积 + 宽高比 + 水平梯度结构
            grad_score = _horizontal_gradient_score(roi_gray, mask_enhanced, x, y, w, h)
            score = float(area) * 0.2 + float(aspect) * 20.0 + grad_score * 3.5

            candidates.append({
                "bbox": [int(x), int(y), int(w), int(h)],
                "center_x": float(center_x),
                "center_y": float(center_y),
                "area": int(area),
                "aspect": float(aspect),
                "score": score
            })

        if len(candidates) < 2:
            return []

    elif len(candidates) < 2:
        return []

    # 按综合评分降序排列
    candidates.sort(key=lambda item: item["score"], reverse=True)

    # Y 轴方向去重（防止同条线被重复提取）
    selected = []
    min_gap = roi_h * float(options.get("red_line_min_gap_ratio", 0.08))

    for item in candidates:
        if all(abs(item["center_y"] - s["center_y"]) > min_gap for s in selected):
            selected.append(item)
        if len(selected) >= 6:
            break

    if len(selected) < 2:
        return []

    # 按中心 Y 轴升序排列，选取最上和最下的两条红线
    selected.sort(key=lambda item: item["center_y"])
    return [selected[0], selected[-1]]


# ===========================================================================
#  第二段步骤三：油面高度定位
# ===========================================================================

def _find_oil_level(
    roi: np.ndarray,
    tube: dict[str, int],
    options: dict[str, Any],
) -> dict[str, float] | None:
    """
    在液柱管体内部定位油面高度。

    【修改说明】搜索区域从"上下红线之间"改为"整个管体区域"。
    即在整个管体高度范围内搜索油面位置，不再依赖红线位置来限定搜索范围。

    算法原理：
      液柱管体内，油面以上是空气（较亮），油面以下是油液（较暗或颜色不同）。
      这个亮度的突变形成了一个强烈的水平梯度。
      通过取管体中心竖直条带，计算每行的平均亮度，再对亮度曲线求一阶差分，
      差分最大处就是油面位置。

    步骤：
      1. 从 ROI 中裁出管体区域（左右由 tube 边界决定）
      2. 转灰度，取中心 60% 的竖直条带（避免管壁边缘干扰）
      3. 高斯模糊（去噪）→ 按行求平均亮度 → 得到亮度剖面曲线
      4. 对亮度曲线求一阶差分（梯度）
      5. 在整个管体高度范围内，找到梯度最大处，即为油面

    参数:
        roi:     油位表区域的小图
        tube:    管体边界 {"left", "right", ...}
        options: 算法参数
            - tube_inner_left_ratio:  中心条带左边界比例（默认 0.2）
            - tube_inner_right_ratio: 中心条带右边界比例（默认 0.8）
            - level_min_score:        最低梯度得分阈值（默认 1.5）

    返回:
        成功时返回 {"y": int, "score": float}
          - y:     油面在 ROI 中的 y 坐标
          - score: 梯度得分（越高说明油面越明显）
        失败时返回 None
    """
    left = tube["left"]
    right = tube["right"]
    # 裁出管体区域
    tube_img = roi[:, left:right]
    if tube_img.shape[1] < 20:
        return None  # 管体太窄

    # 转灰度，取中心竖直条带（避免管壁边缘的梯度干扰）
    gray = cv2.cvtColor(tube_img, cv2.COLOR_BGR2GRAY)
    inner_left = int(gray.shape[1] * float(options.get("tube_inner_left_ratio", 0.2)))
    inner_right = int(gray.shape[1] * float(options.get("tube_inner_right_ratio", 0.8)))
    center_strip = gray[:, inner_left:inner_right]
    if center_strip.size == 0:
        return None

    # 高斯模糊去噪（ksize=(1,9) 表示只在水平方向平噪，保留垂直方向的油面突变）
    blurred = cv2.GaussianBlur(center_strip, (1, 9), 0)

    # 按行求平均亮度 → 得到一维亮度剖面（每行一个值）
    profile = blurred.mean(axis=1)

    # 对亮度剖面求一阶差分（相邻行的亮度差异）
    # 油面处亮度突变，差分值最大
    grad = np.abs(np.diff(profile))
    if grad.size == 0:
        return None

    # 在管体下 72% 范围内搜索油面（舍弃上 28%）
    # 设置油面搜索范围：跳过上 28% 和下 30%（即仅在中间 28% ~ 70% 区域内搜索）
    total_len = len(grad)
    search_top = int(total_len * 0.28)  # 跳过顶部 28%
    search_bottom = int(total_len * 0.70) - 1  # 跳过底部 30%（取上限为 70%）

    if search_bottom <= search_top:
        return None

    # 在整个管体范围内找梯度最大处 → 即为油面
    level_idx = int(np.argmax(grad[search_top : search_bottom + 1]) + search_top)
    score = float(grad[level_idx])

    # 梯度得分低于阈值，说明没有明显的油面突变，证据不足
    min_score = float(options.get("level_min_score", 1.5))
    if score < min_score:
        return None

    return {
        "y": level_idx,
        "score": score,
    }


# ===========================================================================
#  坐标转换辅助函数
# ===========================================================================

def _abs_tube_bounds(tube: dict[str, int], bbox_xyxy: list[int]) -> list[int]:
    """
    将管体边界从 ROI 内坐标转换为原图绝对坐标。

    参数:
        tube:      管体边界 {"left", "top", "right", "bottom"}（ROI 内坐标）
        bbox_xyxy: 检测框在原图中的位置 [x1, y1, x2, y2]

    返回:
        [x1_abs, y1_abs, x2_abs, y2_abs] 原图绝对坐标
    """
    x1, y1 = bbox_xyxy[0], bbox_xyxy[1]
    return [
        int(x1 + tube["left"]),
        int(y1 + tube["top"]),
        int(x1 + tube["right"]),
        int(y1 + tube["bottom"]),
    ]


def _abs_line_positions(lines: list[dict[str, Any]], bbox_xyxy: list[int]) -> list[int]:
    """
    将红线 y 坐标从 ROI 内坐标转换为原图绝对坐标。

    参数:
        lines:     红线列表 [{"center_y": float}, ...]
        bbox_xyxy: 检测框在原图中的位置

    返回:
        [y_upper_abs, y_lower_abs]
    """
    y1 = bbox_xyxy[1]
    return [int(y1 + line["center_y"]) for line in lines]


# ===========================================================================
#  可视化绘制函数
# ===========================================================================

def _draw_detection_only(image: np.ndarray, bbox_xyxy: list[int], confidence: float) -> np.ndarray:
    """
    绘制仅有 YOLO 检测框的标注图（管体定位失败时使用）。

    在原图上画一个黄色矩形框，并标注置信度。
    """
    output = image.copy()
    x1, y1, x2, y2 = bbox_xyxy
    # 黄色矩形框 (BGR: 0,255,255)
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 2)
    # 在框上方标注置信度
    cv2.putText(
        output,
        f"gauge conf={confidence:.2f}",
        (x1, max(24, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def _draw_structure_result(
    image: np.ndarray,
    bbox_xyxy: list[int],
    confidence: float,
    tube: dict[str, int],
    lines: list[dict[str, Any]],
    level: dict[str, float] | None,
    state: str,
) -> np.ndarray:
    """
    绘制完整的结构化标注结果图。

    标注内容：
      - 油位表检测框（颜色随状态变化：绿色=正常，红色=异常，黄色=未知）
      - 液柱管体边界（青色）
      - 两条红色参考线（红色水平线，仅在管体左侧区域绘制）
      - 油面位置（绿色水平线，如果找到）
      - 左上角状态标签栏

    参数:
        image:     原图
        bbox_xyxy: 检测框坐标
        confidence:YOLO 置信度
        tube:      管体边界
        lines:     红线列表
        level:     油面信息（可为 None）
        state:     当前状态 "normal" / "abnormal" / "unknown"
    """
    output = image.copy()
    x1, y1, x2, y2 = bbox_xyxy

    # 颜色方案：正常=绿色，异常=红色，未知=黄色
    state_color = (0, 180, 0) if state == "normal" else (0, 0, 255) if state == "abnormal" else (0, 255, 255)

    # 画油位表检测框
    cv2.rectangle(output, (x1, y1), (x2, y2), state_color, 2)
    # 画管体边界（青色矩形）
    cv2.rectangle(output, (x1 + tube["left"], y1), (x1 + tube["right"], y2 - 1), (255, 255, 0), 2)

    # 画红色参考线（水平红线，仅在管体左侧区域）
    for item in lines:
        ly = int(y1 + item["center_y"])
        # 只画到管体左边界
        cv2.line(output, (x1, ly), (x1 + tube["left"], ly), (0, 0, 255), 2, cv2.LINE_AA)

    # 画油面位置（绿色水平线）
    if level is not None:
        oil_y = int(y1 + level["y"])
        cv2.line(output, (x1 + tube["left"], oil_y), (x1 + tube["right"], oil_y), (0, 255, 0), 2, cv2.LINE_AA)

    # 左上角标签栏（黑色背景 + 状态文字）
    oil_text = "oil=unknown" if level is None else f"oil_y={int(y1 + level['y'])} score={level['score']:.2f}"
    label = f"{state} conf={confidence:.2f} {oil_text}"
    cv2.rectangle(output, (10, 10), (min(output.shape[1] - 1, 820), 56), (0, 0, 0), -1)
    cv2.putText(output, label, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, state_color, 2, cv2.LINE_AA)
    return output


def _draw_unknown_banner(image: np.ndarray, reason: str) -> np.ndarray:
    """
    在图片左上角绘制 UNKNOWN 状态的提示横幅。

    用于流水线失败时，标注失败原因。
    """
    output = image.copy()
    # 黑色背景横幅
    cv2.rectangle(output, (10, 10), (min(output.shape[1] - 1, 720), 56), (0, 0, 0), -1)
    # 黄色文字标注原因
    cv2.putText(
        output,
        f"oil_level: unknown {reason}",
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output
