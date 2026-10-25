#!/usr/bin/env python3
"""just tests"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import sys
import types
import uuid
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VALID_IMAGE = PROJECT_ROOT / "test_valid.png"
INVALID_IMAGE = PROJECT_ROOT / "test_invalid.txt"


def _load_vulnerable_module() -> types.ModuleType:
    """Load the legacy script under __main__ so its supplied pickles resolve"""

    import flask

    script_path = PROJECT_ROOT / "lab_access_api.py"
    original_main = sys.modules["__main__"]
    original_run = flask.Flask.run
    module = types.ModuleType("__main__")
    module.__file__ = str(script_path)
    module.__package__ = None
    sys.modules["__main__"] = module
    flask.Flask.run = lambda self, *args, **kwargs: None
    try:
        spec = importlib.util.spec_from_file_location("__main__", script_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("could not create legacy module spec")
        spec.loader.exec_module(module)
        return module
    finally:
        flask.Flask.run = original_run
        sys.modules["__main__"] = original_main


def _response_record(response: Any) -> dict[str, Any]:
    body = response.get_data(as_text=True)
    return {
        "status": response.status_code,
        "content_type": response.headers.get("Content-Type"),
        "json": response.get_json(silent=True),
        "body_length": len(body),
        "body_excerpt": body[:6000],
    }


def _post_image(client: Any, image_path: Path, filename: str | None = None) -> dict[str, Any]:
    image_name = filename or image_path.name
    with image_path.open("rb") as image_handle:
        response = client.post(
            "/api/access",
            data={"image": (io.BytesIO(image_handle.read()), image_name)},
            content_type="multipart/form-data",
        )
    return _response_record(response)


def _post_bytes(client: Any, payload: bytes, filename: str) -> dict[str, Any]:
    response = client.post(
        "/api/access",
        data={"image": (io.BytesIO(payload), filename)},
        content_type="multipart/form-data",
    )
    return _response_record(response)


def _baseline_cases(module: types.ModuleType) -> dict[str, Any]:
    client = module.app.test_client()
    cases: dict[str, Any] = {}
    cases["A_no_image"] = _response_record(client.post("/api/access"))
    cases["B_valid_image"] = _post_image(client, VALID_IMAGE)
    cases["C_non_image"] = _post_image(client, INVALID_IMAGE)

    raw_preprocessed = module.preprocess_image(VALID_IMAGE.read_bytes())
    cases["preprocessing_observation"] = {
        "shape": list(raw_preprocessed.shape) if raw_preprocessed is not None else None,
        "dtype": str(raw_preprocessed.dtype) if raw_preprocessed is not None else None,
        "decoded": raw_preprocessed is not None,
    }
    try:
        module.face_model.predict(raw_preprocessed)
    except Exception as exc:
        cases["model_shape_rejection"] = {
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }

    # Isolate the later policy/response path from the consumed-stream bug by
    # supplying the expected tensort his is a controlled instrumentation,
    # not a claim that the original endpoint can naturally reach this path
    original_preprocess = module.preprocess_image
    module.preprocess_image = lambda _: np.zeros((1, 224, 224, 3), dtype=np.float32)
    try:
        isolated_name = f"baseline_isolated_{uuid.uuid4().hex}.png"
        cases["isolated_policy_and_response"] = _post_image(
            client, VALID_IMAGE, filename=isolated_name
        )
        traversal_name = f"../logs/baseline_controlled_{uuid.uuid4().hex}.png"
        traversal_result = _post_image(client, VALID_IMAGE, filename=traversal_name)
        traversal_path = (PROJECT_ROOT / "logs" / ".." / "logs" / Path(traversal_name).name).resolve()
        traversal_result["controlled_in_lab_path_exists"] = traversal_path.exists()
        traversal_result["outside_lab_attempted"] = False
        cases["controlled_filename_path_test"] = traversal_result
    finally:
        module.preprocess_image = original_preprocess

    cases["configuration_observation"] = {
        "max_content_length": module.app.config.get("MAX_CONTENT_LENGTH"),
        "debug": module.app.debug,
        "source_file": str(PROJECT_ROOT / "lab_access_api.py"),
    }
    return cases


def _secure_cases(module: types.ModuleType) -> dict[str, Any]:
    client = module.app.test_client()
    cases: dict[str, Any] = {}
    cases["A_no_image"] = _response_record(client.post("/api/access"))
    cases["B_valid_image"] = _post_image(client, VALID_IMAGE)
    cases["C_non_image"] = _post_image(client, INVALID_IMAGE)

    oversized = b"x" * (module.MAX_REQUEST_BYTES + 1)
    cases["D_oversized_request"] = _post_bytes(client, oversized, "oversized.bin")

    tensor = module.preprocess_image(VALID_IMAGE.read_bytes())
    cases["preprocessing_observation"] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "minimum": float(tensor.min()),
        "maximum": float(tensor.max()),
    }
    cases["model_loading_observation"] = {
        "face_model_type": type(module.face_model).__name__,
        "object_model_type": type(module.object_model).__name__,
        "face_model_sha256": module.EXPECTED_MODEL_SHA256["face_recognition_model.pkl"],
        "object_model_sha256": module.EXPECTED_MODEL_SHA256["object_detection_model.pkl"],
    }

    upload_dir = module.UPLOAD_DIR
    before = {path.name for path in upload_dir.iterdir() if path.is_file()}
    traversal_result = _post_image(client, VALID_IMAGE, filename="../../outside-lab.png")
    after = {path.name for path in upload_dir.iterdir() if path.is_file()}
    new_files = sorted(after - before)
    traversal_result["generated_files"] = new_files
    traversal_result["all_generated_files_direct_children"] = all(
        (upload_dir / name).resolve().parent == upload_dir.resolve() for name in new_files
    )
    traversal_result["outside_lab_attempted"] = False
    cases["controlled_filename_path_test"] = traversal_result

    cases["configuration_observation"] = {
        "max_content_length": module.app.config.get("MAX_CONTENT_LENGTH"),
        "debug": module.app.debug,
        "upload_dir": str(module.UPLOAD_DIR),
    }
    return cases


def _unique_output(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError("could not choose a non-overwriting evidence path")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=("vulnerable", "secure"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.target == "vulnerable":
        module = _load_vulnerable_module()
        cases = _baseline_cases(module)
    else:
        import lab_access_api_secure as module

        cases = _secure_cases(module)

    output_path = _unique_output((PROJECT_ROOT / args.output).resolve())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "target": args.target,
        "execution": "local Flask test_client no external network target",
        "working_directory": str(PROJECT_ROOT),
        "cases": cases,
    }
    output_path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"evidence": str(output_path), "target": args.target}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
