"""
vision/inference.py — Lightweight ONNX inference for TBScreenAI chest X-ray screening.

Stripped down from the full TBScreenAI edge deployment engine. No TensorFlow,
no Grad-CAM, no TFLite — just ONNX Runtime on CPU. Designed to fit within
the ADTC 7GB RAM ceiling alongside the LLM.

The model is MobileNetV3-Small trained via 3-stage transfer learning on
TBX11K (11,200 images), Shenzhen (662 images), OpenI (7,056 images), and
NIH ChestX-ray14 (112,120 images). Validation AUC: 0.930, Test AUC: 0.888.

Preprocessing must match training exactly:
  1. Convert to RGB
  2. CLAHE on grayscale (clip_limit=2.0, tile_grid_size=8x8)
  3. Resize to 224x224 (bilinear)
  4. Float32, divide by 255
  5. ImageNet normalisation (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
"""

import io
import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_SIZE = 224
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Production screening threshold (v1.1 evaluation, t=0.65)
# Sensitivity: 70.2%, FPR: 11.7%
SCREENING_THRESHOLD = 0.65


def _apply_clahe(pil_image: Image.Image) -> Image.Image:
    """Apply CLAHE to enhance local contrast — must match training preprocessing."""
    import cv2
    gray = np.array(pil_image.convert("L"), dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return Image.fromarray(enhanced, mode="L").convert("RGB")


def preprocess(image_path: str) -> np.ndarray:
    """
    Preprocess a chest X-ray image for inference.

    Returns float32 array of shape (1, 224, 224, 3).
    """
    img = Image.open(image_path).convert("RGB")
    img = _apply_clahe(img)
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return np.expand_dims(arr, axis=0)


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class TBScreenModel:
    """
    ONNX-based TB screening model.

    Usage:
        model = TBScreenModel("vision/model/tb_model.onnx")
        result = model.predict("path/to/xray.jpg")
        print(result)
        # {
        #     "tb_probability": 0.783,
        #     "prediction": "TB-positive",
        #     "confidence": 0.783,
        #     "screening_result": "POSITIVE",
        #     "threshold": 0.65
        # }
    """

    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = str(Path(__file__).parent / "model" / "tb_model.onnx")

        import onnxruntime as ort
        self._session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        logger.info("TBScreenAI ONNX model loaded: %s", model_path)

    def predict(self, image_path: str, threshold: float = SCREENING_THRESHOLD) -> dict:
        """
        Screen a chest X-ray for TB.

        Parameters
        ----------
        image_path : str
            Path to the X-ray image file.
        threshold : float
            Screening threshold. Default 0.65 (production calibrated).

        Returns
        -------
        dict with keys:
            tb_probability   : float (0-1)
            prediction       : str ("TB-positive" or "TB-negative")
            confidence       : float (0-1, confidence in the prediction)
            screening_result : str ("POSITIVE" or "NEGATIVE")
            threshold        : float (threshold used)
        """
        preprocessed = preprocess(image_path)
        outputs = self._session.run(None, {self._input_name: preprocessed})
        probs = outputs[0].astype(np.float32)

        if abs(probs.sum() - 1.0) > 0.05:
            probs = softmax(probs)

        tb_prob = float(probs[0][1])  # index 1 = TB-positive class
        is_positive = tb_prob >= threshold

        return {
            "tb_probability": round(tb_prob, 4),
            "prediction": "TB-positive" if is_positive else "TB-negative",
            "confidence": round(tb_prob if is_positive else 1 - tb_prob, 4),
            "screening_result": "POSITIVE" if is_positive else "NEGATIVE",
            "threshold": threshold,
        }
