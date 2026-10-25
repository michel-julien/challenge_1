#!/usr/bin/env python3
"""Create a compact before/after comparison from two runs"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _status(document: dict[str, Any], case: str) -> int | None:
    value = document.get("cases", {}).get(case, {}).get("status")
    return int(value) if value is not None else None


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
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--secure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_path = (ROOT / args.baseline).resolve()
    secure_path = (ROOT / args.secure).resolve()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    secure = json.loads(secure_path.read_text(encoding="utf-8"))
    cases = [
        "A_no_image",
        "B_valid_image",
        "C_non_image",
        "D_oversized_request",
        "controlled_filename_path_test",
    ]
    comparison = {
        "execution": "derived only from confirmed local evidence JSON",
        "baseline_file": str(baseline_path),
        "secure_file": str(secure_path),
        "status_by_case": {
            case: {
                "baseline": _status(baseline, case),
                "secure": _status(secure, case),
            }
            for case in cases
        },
        "observations": {
            "baseline_preprocessing": baseline.get("cases", {}).get("preprocessing_observation"),
            "secure_preprocessing": secure.get("cases", {}).get("preprocessing_observation"),
            "baseline_isolated_response": baseline.get("cases", {}).get("isolated_policy_and_response", {}).get("json"),
            "secure_valid_response": secure.get("cases", {}).get("B_valid_image", {}).get("json"),
            "secure_model_loading": secure.get("cases", {}).get("model_loading_observation"),
        },
    }
    output_path = _unique_output((ROOT / args.output).resolve())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"evidence": str(output_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
