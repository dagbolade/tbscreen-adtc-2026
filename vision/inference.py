"""
vision/inference.py — Lightweight ONNX inference for TBScreenAI chest X-ray screening.

Stripped down from the full TBScreenAI edge deployment engine. No TensorFlow,
no TFLite — ONNX Runtime on CPU plus optional occlusion zone probes for the
LLM integration layer. Designed to fit within the ADTC 7GB RAM ceiling.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_SIZE = 224
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Production screening threshold (v1.1 evaluation, t=0.65)
SCREENING_THRESHOLD = 0.65

# Anatomical-ish 2x2 grid used for occlusion zone activations.
ZONE_GRID = {
    "upper_left": (0.0, 0.0, 0.5, 0.5),
    "upper_right": (0.5, 0.0, 1.0, 0.5),
    "lower_left": (0.0, 0.5, 0.5, 1.0),
    "lower_right": (0.5, 0.5, 1.0, 1.0),
}


def _apply_clahe(pil_image: Image.Image) -> Image.Image:
    """Apply CLAHE to enhance local contrast — must match training preprocessing."""
    import cv2

    gray = np.array(pil_image.convert("L"), dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return Image.fromarray(enhanced, mode="L").convert("RGB")


def _to_batch(pil_image: Image.Image) -> np.ndarray:
    """CLAHE → resize → ImageNet-normalize → NHWC batch."""
    img = _apply_clahe(pil_image.convert("RGB"))
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return np.expand_dims(arr, axis=0)


def preprocess(image_path: str) -> np.ndarray:
    """Preprocess a chest X-ray; returns float32 array of shape (1, 224, 224, 3)."""
    return _to_batch(Image.open(image_path).convert("RGB"))


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _tb_prob_from_outputs(outputs: list[np.ndarray]) -> float:
    probs = outputs[0].astype(np.float32)
    if abs(float(probs.sum()) - 1.0) > 0.05:
        probs = softmax(probs)
    return float(probs[0][1])


class TBScreenModel:
    """ONNX TB screener with optional 2x2 occlusion zone activations for the LLM."""

    def __init__(self, model_path: str | None = None):
        if model_path is None:
            model_path = str(Path(__file__).parent / "model" / "tb_model.onnx")

        import onnxruntime as ort

        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        logger.info("TBScreenAI ONNX model loaded: %s", model_path)

    def _forward(self, batch: np.ndarray) -> float:
        outputs = self._session.run(None, {self._input_name: batch})
        return _tb_prob_from_outputs(outputs)

    def _zone_activations(self, pil_image: Image.Image, baseline: float) -> dict[str, float]:
        """Occlusion sensitivity: drop in TB prob when each quadrant is zeroed."""
        w, h = pil_image.size
        drops: dict[str, float] = {}
        for name, (x0, y0, x1, y1) in ZONE_GRID.items():
            occluded = pil_image.copy()
            box = (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))
            black = Image.new("RGB", (box[2] - box[0], box[3] - box[1]), (0, 0, 0))
            occluded.paste(black, box[:2])
            occluded_prob = self._forward(_to_batch(occluded))
            drops[name] = max(0.0, baseline - occluded_prob)

        total = sum(drops.values())
        if total < 1e-8:
            # Uniform fallback when occlusion does not move the score.
            return {k: round(1.0 / len(ZONE_GRID), 4) for k in ZONE_GRID}
        return {k: round(v / total, 4) for k, v in drops.items()}

    def predict(
        self,
        image_path: str,
        threshold: float = SCREENING_THRESHOLD,
        with_zones: bool = True,
    ) -> dict:
        """Screen a chest X-ray; optionally attach normalized zone activation weights."""
        pil = Image.open(image_path).convert("RGB")
        tb_prob = self._forward(_to_batch(pil))
        is_positive = tb_prob >= threshold

        result = {
            "tb_probability": round(tb_prob, 4),
            "prediction": "TB-positive" if is_positive else "TB-negative",
            "confidence": round(tb_prob if is_positive else 1 - tb_prob, 4),
            "screening_result": "POSITIVE" if is_positive else "NEGATIVE",
            "threshold": threshold,
        }
        if with_zones:
            zones = self._zone_activations(pil, tb_prob)
            result["zone_activations"] = zones
            # Dominant zone helps the LLM localize the finding in plain language.
            result["dominant_zone"] = max(zones, key=zones.get)
        return result
