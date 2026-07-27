"""
油位表检测服务 - API 层

本模块是油位表识别系统的 HTTP 服务入口，基于 FastAPI 框架构建。
负责以下职责：
  1. 从 MinIO 对象存储读取待检测的图片
  2. 调用 OilLevelGaugeDetector 执行油位识别
  3. 将标注了检测结果的图片上传回 MinIO
  4. 返回 JSON 格式的检测结果（包含临时访问 URL）

接口说明：
  - GET  /health       → 健康检查，返回模型加载状态
  - POST /detect/minio → 批量油位检测，接收 MinIO 图片 key 列表

MinIO 存储约定：
  - 待检测图片路径：beforeDetected/<filename>
  - 结果图片路径：  afterDetected/<filename>-oil-level-result.jpg

启动方式：
  直接运行本文件：python api_minio.py
  或通过 uvicorn：uvicorn api_minio:app --host 0.0.0.0 --port 8700
"""

from __future__ import annotations

import io
import os
from datetime import timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from minio import Minio
from pydantic import BaseModel, Field

# 导入油位表检测算法核心类
from oil_level_gauge_algorithm import OilLevelGaugeDetector


# ===========================================================================
#  配置项（通过环境变量覆盖，均有默认值）
# ===========================================================================

# MinIO 服务连接地址（格式：host:port）
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
# MinIO 访问凭证
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
# MinIO 存储桶名称
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "rail-robot")
# 待检测图片在 MinIO 中的路径前缀
MINIO_INPUT_PREFIX = os.getenv("MINIO_INPUT_PREFIX", "beforeDetected")
# 结果图片在 MinIO 中的路径前缀
MINIO_OUTPUT_PREFIX = os.getenv("MINIO_OUTPUT_PREFIX", "afterDetected")
# 是否使用 HTTPS 连接 MinIO
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() in {"1", "true", "yes"}

# YOLO 模型权重文件路径（相对于本文件所在目录）
MODEL_PATH = Path(__file__).resolve().parent / "weights" / "best.pt"


# ===========================================================================
#  默认算法参数
# ===========================================================================
# 这些参数会被传入 OilLevelGaugeDetector，作为各步骤的默认阈值。
# 请求时可以通过 params 字段覆盖。
#
# 【已更新】与算法侧保持一致：
#   - 红色参考线仅在管体左边界左侧区域搜索（red_side_band_ratio 为管壁容差）
#   - 油面搜索区域从"上下红线之间"改为"整个管体区域"
#     → 已移除 level_search_top_margin_ratio、level_search_bottom_margin_ratio

DEFAULT_OPTIONS: dict[str, Any] = {
    # === YOLO 检测参数 ===
    "det_conf": 0.25,           # 检测置信度阈值，低于此值的检测框被过滤
    "det_iou": 0.45,            # NMS（非极大值抑制）IoU 阈值

    # === 管体定位参数 ===
    "tube_search_left_ratio": 0.15,     # 搜索范围左边界（占 ROI 宽度的比例）
    "tube_search_right_ratio": 0.85,    # 搜索范围右边界
    "tube_min_width_ratio": 0.08,       # 管体最小宽度比例
    "tube_max_width_ratio": 0.45,       # 管体最大宽度比例
    "tube_expected_width_ratio": 0.22,  # 预期宽度比例（用于评分）
    "tube_inner_left_ratio": 0.2,       # 油面搜索时取管体中心区域的左边界
    "tube_inner_right_ratio": 0.8,      # 油面搜索时取管体中心区域的右边界

    # === 红色参考线识别参数（仅在管体左边界左侧区域搜索）===
    "red_hue_low_min": 0,       # 红色 HSV 低 H 段下界
    "red_hue_low_max": 12,      # 红色 HSV 低 H 段上界
    "red_hue_high_min": 160,    # 红色 HSV 高 H 段下界
    "red_hue_high_max": 179,    # 红色 HSV 高 H 段上界
    "red_saturation_min": 80,   # 最小饱和度（过低的颜色不够"红"）
    "red_value_min": 80,        # 最小明度（过暗的区域不算）
    "red_min_area_ratio": 0.0005,    # 红色连通域最小面积比例
    "red_min_width_ratio": 0.08,     # 红色连通域最小宽度比例
    "red_min_aspect_ratio": 1.2,     # 最小宽高比（红线是横向的）
    "red_side_band_ratio": 0.18,     # 管体左边界附近的容差带比例
    "red_line_min_gap_ratio": 0.08,  # 两条红线之间最小间距比例

    # === 油面定位参数（在整个管体区域内搜索）===
    "level_min_score": 2.0,          # 油面梯度最低得分阈值
}


# ===========================================================================
#  请求数据模型（Pydantic 校验）
# ===========================================================================

class DetectRequest(BaseModel):
    """
    检测请求体。

    字段:
        object_keys: MinIO 中待检测图片的对象键列表。
                     支持两种格式：
                       - 完整路径: "beforeDetected/xxx.jpg"
                       - 仅文件名: "xxx.jpg"（会自动拼接 beforeDetected/ 前缀）
        params:      可选的算法参数覆盖字典，会与 DEFAULT_OPTIONS 合并。
    """
    object_keys: list[str] = Field(..., description="MinIO 图片对象键，支持带 beforeDetected/ 前缀或只传文件名")
    params: dict[str, Any] = Field(default_factory=dict, description="算法阈值覆盖")


# ===========================================================================
#  FastAPI 应用初始化
# ===========================================================================

# 创建 FastAPI 应用实例
app = FastAPI(title="油位表检测服务", version="1.0.0")

# 初始化 MinIO 客户端（用于读取输入图片和上传结果图片）
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=MINIO_SECURE,
)

# 初始化油位表检测器，加载 YOLO 模型
# 注意：这会在服务启动时就加载模型到内存，首次请求无需等待模型加载
detector = OilLevelGaugeDetector(MODEL_PATH, defaults=DEFAULT_OPTIONS)
detector.load()


# ===========================================================================
#  API 路由
# ===========================================================================

@app.get("/health")
def health() -> dict[str, Any]:
    """
    健康检查接口。

    返回:
        {"status": "ok", "model_loaded": true/false}
    """
    return {"status": "ok", "model_loaded": detector.loaded}


@app.post("/detect/minio")
def detect_from_minio(request: DetectRequest) -> dict[str, Any]:
    """
    批量油位检测接口（从 MinIO 读取图片）。

    工作流程：
      1. 接收一批 MinIO 对象键
      2. 逐个从 MinIO 读取图片，执行油位识别
      3. 将标注结果图上传回 MinIO，生成临时访问 URL
      4. 汇总所有结果返回

    参数:
        request: DetectRequest，包含 object_keys 列表和可选的 params

    返回:
        {
          "final_check_result": "NORMAL" | "ABNORMAL" | "UNKNOWN" | "ALL_READ_FAILED",
          "details": [
            {
              "original_object_key": 原始请求的 key,
              "source_object_key": 实际读取的 key（规范化后）,
              "detected": true/false,
              "check_result": "NORMAL"/"ABNORMAL"/"UNKNOWN"/"READ_FAILED",
              "state": "normal"/"abnormal"/"unknown",
              "is_normal": true/false,
              "reason": 判定原因,
              "confidence": 置信度,
              "gauge_bbox": 检测框,
              "tube_bounds": 管体边界,
              "reference_lines_y": 红线 y 坐标,
              "oil_level_y": 油面 y 坐标,
              "oil_level_position_ratio": 油面相对位置,
              "oil_level_score": 油面梯度得分,
              "cost_ms": 耗时毫秒,
              "result_img_objectKey": 结果图的 MinIO key,
              "result_img_objectUrl": 结果图的临时访问 URL
            },
            ...
          ]
        }
    """
    if not request.object_keys:
        raise HTTPException(status_code=400, detail="object_keys 不能为空")

    # 合并默认参数和请求参数
    options = dict(DEFAULT_OPTIONS)
    options.update(request.params or {})

    # 逐张处理（串行，适用于小批量场景）
    details = [_detect_one(object_key, options) for object_key in request.object_keys]
    return {
        "final_check_result": _final_result(details),  # 汇总最终结果
        "details": details,
    }


# ===========================================================================
#  内部处理函数
# ===========================================================================

def _detect_one(object_key: str, options: dict[str, Any]) -> dict[str, Any]:
    """
    对单张图片执行完整检测流程：读取 → 推理 → 上传结果。

    参数:
        object_key: MinIO 中的图片对象键
        options:    算法参数字典

    返回:
        包含检测结果和结果图 URL 的字典。
        如果图片读取失败，返回 READ_FAILED 状态。
    """
    # 规范化输入 key（自动补 beforeDetected/ 前缀）
    source_key = _normalize_input_key(object_key)

    try:
        # 从 MinIO 读取图片字节数据
        img_data = _read_minio_object(source_key)
        # 将字节数据解码为 OpenCV 图片（BGR numpy 数组）
        image = _imread_from_bytes(img_data)
    except Exception as exc:
        # 图片读取或解码失败，返回 READ_FAILED 状态
        return {
            "original_object_key": object_key,
            "source_object_key": source_key,
            "detected": False,
            "check_result": "READ_FAILED",
            "state": "unknown",
            "is_normal": False,
            "reason": str(exc),
            "confidence": 0.0,
            "gauge_bbox": [],
            "tube_bounds": [],
            "reference_lines_y": [],
            "oil_level_y": None,
            "oil_level_position_ratio": None,
            "oil_level_score": 0.0,
            "result_img_objectKey": "",
            "result_img_objectUrl": "",
        }

    # 调用核心算法执行油位识别
    result = detector.infer(image, options)

    # 将结果图上传到 MinIO
    result_key = _result_key(source_key)
    result_url = _upload_result_image(result_key, result["result_image"])

    # 组装返回结果
    return {
        "original_object_key": object_key,
        "source_object_key": source_key,
        "detected": result["state"] != "unknown",     # 检测是否成功
        "check_result": result["check_result"],
        "state": result["state"],
        "is_normal": result["is_normal"],
        "reason": result["reason"],
        "confidence": result["confidence"],
        "gauge_bbox": result["gauge_bbox"],
        "tube_bounds": result["tube_bounds"],
        "reference_lines_y": result["reference_lines_y"],
        "oil_level_y": result["oil_level_y"],
        "oil_level_position_ratio": result["oil_level_position_ratio"],
        "oil_level_score": result["oil_level_score"],
        "cost_ms": result["cost_ms"],
        "result_img_objectKey": result_key,
        "result_img_objectUrl": result_url,
    }


def _read_minio_object(object_key: str) -> bytes:
    """
    从 MinIO 读取指定对象的全部字节数据。

    参数:
        object_key: 对象在 MinIO 中的完整路径

    返回:
        图片的原始字节数据

    异常:
        MinIO 不存在该对象时会抛出异常
    """
    response = minio_client.get_object(MINIO_BUCKET, object_key)
    try:
        return response.read()
    finally:
        # 确保释放连接资源（MinIO 客户端要求）
        response.close()
        response.release_conn()


def _imread_from_bytes(data: bytes) -> np.ndarray:
    """
    将图片字节数据解码为 OpenCV 格式的 numpy 数组。

    支持所有 OpenCV 支持的图片格式（JPEG、PNG、BMP 等）。

    参数:
        data: 图片的原始字节数据

    返回:
        BGR 格式的 numpy 数组 (H, W, 3)

    异常:
        解码失败时抛出 ValueError
    """
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("图片解码失败")
    return image


def _upload_result_image(object_key: str, image: np.ndarray) -> str:
    """
    将检测结果图片上传到 MinIO，并返回带签名的临时访问 URL。

    流程：
      1. 将 numpy 数组编码为 JPEG 字节
      2. 上传到 MinIO 的指定路径
      3. 生成 1 小时有效的预签名 URL

    参数:
        object_key: 结果图在 MinIO 中的存储路径
        image:      BGR 格式的结果图片（numpy 数组）

    返回:
        1 小时有效的预签名 GET URL
    """
    # 编码为 JPEG 格式
    ok, buffer = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("结果图编码失败")

    img_bytes = buffer.tobytes()
    # 上传到 MinIO
    minio_client.put_object(
        MINIO_BUCKET,
        object_key,
        io.BytesIO(img_bytes),  # MinIO 客户端需要 file-like 对象
        len(img_bytes),
        content_type="image/jpeg",
    )
    # 生成 1 小时有效的预签名 URL，供前端或调用方直接访问图片
    return minio_client.presigned_get_object(MINIO_BUCKET, object_key, expires=timedelta(hours=1))


def _normalize_input_key(object_key: str) -> str:
    """
    规范化输入对象键，确保带有 beforeDetected/ 前缀。

    调用方可以只传文件名（如 "abc.jpg"），也可以传完整路径
    （如 "beforeDetected/abc.jpg"），本函数统一为完整路径。

    参数:
        object_key: 用户传入的原始对象键

    返回:
        带有 beforeDetected/ 前缀的完整路径
    """
    key = object_key.strip().lstrip("/")
    if key.startswith(f"{MINIO_INPUT_PREFIX}/"):
        return key  # 已经有前缀，直接返回
    return f"{MINIO_INPUT_PREFIX}/{key}"  # 自动拼接前缀


def _result_key(source_key: str) -> str:
    """
    根据输入图片的路径，生成结果图的 MinIO 存储路径。

    命名规则：
      输入：beforeDetected/abc.jpg
      输出：afterDetected/abc-oil-level-result.jpg

    参数:
        source_key: 输入图片的完整 MinIO 路径

    返回:
        结果图的 MinIO 路径
    """
    name = Path(source_key).name         # 提取文件名
    stem = Path(name).stem               # 提取不带扩展名的部分
    return f"{MINIO_OUTPUT_PREFIX}/{stem}-oil-level-result.jpg"


def _final_result(details: list[dict[str, Any]]) -> str:
    """
    根据所有图片的检测结果，汇总最终判定。

    规则：
      - 所有图片都 READ_FAILED → "ALL_READ_FAILED"
      - 有任何一张 ABNORMAL     → "ABNORMAL"（一票否决）
      - 有任何一张 NORMAL       → "NORMAL"
      - 其他情况                → "UNKNOWN"

    参数:
        details: 所有单张图片的检测结果列表

    返回:
        汇总结果字符串
    """
    statuses = [item["check_result"] for item in details]
    if all(status == "READ_FAILED" for status in statuses):
        return "ALL_READ_FAILED"
    if "ABNORMAL" in statuses:
        return "ABNORMAL"   # 只要有一张异常，整体就是异常
    if "NORMAL" in statuses:
        return "NORMAL"
    return "UNKNOWN"


# ===========================================================================
#  服务入口
# ===========================================================================

if __name__ == "__main__":
    import uvicorn

    # 启动 HTTP 服务，监听 0.0.0.0:8700
    uvicorn.run(app, host="0.0.0.0", port=8700)
