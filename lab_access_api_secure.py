"""Hardened local version of the simulated access-control API

This module deliberately keeps the supplied mock model interface so the lab can
compare the vulnerable and defended implementations.  It does not expose
model internals in HTTP responses and fails closed when an input or model
output is ambiguous
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import pickle
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "logs" / "secure_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# The limit covers the complete multipart request, the imagelevel check
# below remains useful when the endpoint is exercised without Flask's parser
MAX_REQUEST_BYTES = 1 * 1024 * 1024
MAX_IMAGE_BYTES = 900 * 1024
IMAGE_SIZE = (224, 224)
FACE_CONFIDENCE_THRESHOLD = 0.90
DANGER_CONFIDENCE_THRESHOLD = 0.80

DANGEROUS_LABELS = {
    "weapon",
    "knife",
    "gun",
    "firearm",
    "pistol",
    "rifle",
    "blade",
    "explosive",
    "bomb",
}

# The simulation emits these two benign classes, an unknown class is treated
# as ambiguous and therefore denied rather than silently allowed
KNOWN_SAFE_LABELS = {"person", "backpack"}

# These hashes are pinned in source code so an attacker cannot replace both a
# local manifest and its model file, they match the supplied model artifacts
EXPECTED_MODEL_SHA256 = {
    "face_recognition_model.pkl": (
        "7ca9b20f377c14942da33d403467e2402418e0c10e8ea660787435c6c681d056"
    ),
    "object_detection_model.pkl": (
        "74fd46269e7bc83e59531f24262968cbfca6273e4055d023a646783e7d30692b"
    ),
}


# These definitions match generate_models.py.  The restricted loader maps the
# legacy __main__ references in the supplied pickle files to these trusted
# classes instead of importing arbitrary modules named by the pickle
class MockFaceRecognitionModel:
    def predict(self, image_tensor: np.ndarray) -> dict[str, Any]:
        if not isinstance(image_tensor, np.ndarray):
            raise TypeError("model input must be a numpy array")
        if image_tensor.shape != (1, 224, 224, 3):
            raise ValueError("model input shape is invalid")
        return {"label": "authorized", "confidence": 0.98}


class MockObjectDetectionModel:
    def predict(self, image_tensor: np.ndarray) -> list[dict[str, Any]]:
        if not isinstance(image_tensor, np.ndarray):
            raise TypeError("model input must be a numpy array")
        if image_tensor.shape != (1, 224, 224, 3):
            raise ValueError("model input shape is invalid")
        return [
            {"label": "person", "confidence": 0.95},
            {"label": "knife", "confidence": 0.88},
            {"label": "backpack", "confidence": 0.75},
        ]


class RestrictedModelUnpickler(pickle.Unpickler):
    """Allow only the two known local model classes

    Pickle is not a general safe serialization format the hash check before
    this loader is mandatory, this allowlist is a second defense-in-depth
    measure for the trusted lab artifacts
    """

    _allowed_modules = {
        "__main__",
        "generate_models",
        "lab_access_api",
        "lab_access_api_secure",
    }

    def find_class(self, module: str, name: str) -> Any:
        if module not in self._allowed_modules:
            raise pickle.UnpicklingError("model references a forbidden module")
        if name == "MockFaceRecognitionModel":
            return MockFaceRecognitionModel
        if name == "MockObjectDetectionModel":
            return MockObjectDetectionModel
        raise pickle.UnpicklingError("model references a forbidden class")


def _load_verified_model(filename: str, expected_type: type[Any]) -> Any:
    path = MODEL_DIR / filename
    try:
        model_bytes = path.read_bytes()
    except OSError as exc:
        raise RuntimeError("required model is unavailable") from exc

    digest = hashlib.sha256(model_bytes).hexdigest()
    if digest != EXPECTED_MODEL_SHA256[filename]:
        raise RuntimeError("model integrity check failed")

    try:
        model = RestrictedModelUnpickler(io.BytesIO(model_bytes)).load()
    except (AttributeError, EOFError, ImportError, IndexError, pickle.UnpicklingError) as exc:
        raise RuntimeError("model could not be loaded safely") from exc

    if not isinstance(model, expected_type) or not callable(getattr(model, "predict", None)):
        raise RuntimeError("model type is not approved")
    return model


face_model = _load_verified_model("face_recognition_model.pkl", MockFaceRecognitionModel)
object_model = _load_verified_model("object_detection_model.pkl", MockObjectDetectionModel)


def _decode_image(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise ValueError("empty image")
    try:
        encoded = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise ValueError("image decoding failed") from exc
    if image is None or image.size == 0:
        raise ValueError("image decoding failed")
    return image


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Decode and normalize an image into the model's required tensor shape"""

    image = _decode_image(image_bytes)
    resized = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.expand_dims(tensor, axis=0)
    if tensor.shape != (1, 224, 224, 3) or not np.isfinite(tensor).all():
        raise ValueError("preprocessed image is invalid")
    return tensor


def _confidence(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return None
    return confidence


def _access_decision(face_prediction: Any, object_prediction: Any) -> bool:
    """Return True only when every security condition is positively met"""

    if not isinstance(face_prediction, dict):
        return False
    if face_prediction.get("label") != "authorized":
        return False
    face_confidence = _confidence(face_prediction.get("confidence"))
    if face_confidence is None or face_confidence < FACE_CONFIDENCE_THRESHOLD:
        return False

    if not isinstance(object_prediction, list) or not object_prediction:
        return False

    for detected in object_prediction:
        if not isinstance(detected, dict):
            return False
        label = detected.get("label")
        if not isinstance(label, str):
            return False
        label = label.strip().lower()
        confidence = _confidence(detected.get("confidence"))
        if confidence is None:
            return False

        confirmed_danger = (
            label in DANGEROUS_LABELS
            and confidence >= DANGER_CONFIDENCE_THRESHOLD
        )
        if confirmed_danger:
            return False
        if label in DANGEROUS_LABELS:
            # A lower-confidence hazard is ambiguous, uncertainty must not
            # grant physical access
            return False
        if label not in KNOWN_SAFE_LABELS:
            return False

    return True


def _stored_upload_path(original_filename: str) -> Path:
    # The client name is used only to choose a harmless extension the actual
    # path is generated server-side and is always a direct child of UPLOAD_DIR
    cleaned_name = secure_filename(original_filename or "")
    suffix = Path(cleaned_name).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        suffix = ".bin"
    generated = f"upload_{uuid.uuid4().hex}{suffix}"
    upload_root = UPLOAD_DIR.resolve()
    candidate = (UPLOAD_DIR / generated).resolve()
    if candidate.parent != upload_root:
        raise RuntimeError("generated upload path escaped its directory")
    return candidate


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.update(
        DEBUG=False,
        TESTING=False,
        MAX_CONTENT_LENGTH=MAX_REQUEST_BYTES,
    )

    # Keep diagnostics in the local process log, none are returned to clients
    app.logger.setLevel(logging.INFO)

    @app.errorhandler(RequestEntityTooLarge)
    def handle_too_large(_: RequestEntityTooLarge):
        return jsonify({"error": "Request too large"}), 413

    @app.errorhandler(HTTPException)
    def handle_http_error(error: HTTPException):
        return jsonify({"error": "Invalid request"}), error.code or 400

    @app.errorhandler(Exception)
    def handle_unexpected(error: Exception):
        app.logger.exception("Unhandled request failure: %s", type(error).__name__)
        return jsonify({"error": "Internal server error"}), 500

    @app.route("/api/access", methods=["POST"])
    def check_access():
        request_id = uuid.uuid4().hex
        upload = request.files.get("image")
        if upload is None or not upload.filename:
            return jsonify({"error": "No image provided"}), 400

        try:
            image_bytes = upload.read(MAX_IMAGE_BYTES + 1)
        except (OSError, ValueError):
            return jsonify({"error": "Invalid image file"}), 400

        if len(image_bytes) > MAX_IMAGE_BYTES:
            return jsonify({"error": "Image too large"}), 413

        try:
            image_tensor = preprocess_image(image_bytes)
        except (ValueError, cv2.error):
            return jsonify({"error": "Invalid image file"}), 400

        try:
            stored_path = _stored_upload_path(upload.filename)
            stored_path.write_bytes(image_bytes)
        except OSError:
            app.logger.exception("Could not store validated upload request_id=%s", request_id)
            return jsonify({"error": "Access check unavailable"}), 503
        except RuntimeError:
            app.logger.exception("Unsafe upload path request_id=%s", request_id)
            return jsonify({"error": "Access check unavailable"}), 503

        try:
            face_prediction = face_model.predict(image_tensor)
            object_prediction = object_model.predict(image_tensor)
            access_granted = _access_decision(face_prediction, object_prediction)
        except Exception as exc:  # model errors fail closed and stay server-side
            app.logger.exception("Model failure request_id=%s type=%s", request_id, type(exc).__name__)
            return jsonify({"error": "Access check unavailable"}), 503

        app.logger.info("Access decision request_id=%s access=%s", request_id, access_granted)
        return jsonify({"access": access_granted})

    return app


app = create_app()


if __name__ == "__main__":
    # Local-only by default debug is explicitly disabled
    host = os.environ.get("ACCESS_API_HOST", "127.0.0.1")
    port = int(os.environ.get("ACCESS_API_PORT", "5001"))
    app.run(host=host, port=port, debug=False)
