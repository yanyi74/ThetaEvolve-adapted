from typing import Any, Optional, Tuple

import numpy as np


DEFAULT_ERROR_RETURN = 1e8
TARGET_C5 = 0.38080


def _coerce_to_sequence(solution_data: Any) -> Tuple[Optional[list[float]], str]:
    if isinstance(solution_data, dict):
        for key in ("height_sequence_1", "heights", "height_sequence", "sequence", "data"):
            if key in solution_data:
                solution_data = solution_data[key]
                break
        else:
            return None, "Dictionary input must contain heights or data"

    if isinstance(solution_data, np.ndarray):
        solution_data = solution_data.tolist()
    if not isinstance(solution_data, list):
        return None, "Solution must be a list of floats"
    if not solution_data:
        return None, "Sequence is empty"

    sequence = []
    for item in solution_data:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None, "Sequence contains a non-numeric value"
        if np.isnan(item) or np.isinf(item):
            return None, "Sequence contains NaN or infinity"
        sequence.append(float(item))
    return sequence, ""


def validate_solution(solution_data: Any) -> Tuple[bool, str]:
    sequence, error = _coerce_to_sequence(solution_data)
    if sequence is None:
        return False, error
    arr = np.asarray(sequence, dtype=float)
    if np.any(arr < 0) or np.any(arr > 1):
        return False, "Erdos sequence must be within [0, 1]"
    if abs(float(np.sum(arr)) - len(arr) / 2.0) > 1e-6:
        return False, "Erdos sequence must sum to n/2"
    return True, "Valid Erdos sequence"


def compute_objective_value(solution_data: Any) -> float:
    sequence, error = _coerce_to_sequence(solution_data)
    if sequence is None:
        print(f"Invalid Erdos sequence: {error}")
        return DEFAULT_ERROR_RETURN
    arr = np.asarray(sequence, dtype=float)
    if np.any(arr < 0) or np.any(arr > 1):
        return DEFAULT_ERROR_RETURN
    total = float(np.sum(arr))
    if abs(total - len(arr) / 2.0) > 1e-6:
        arr = arr * ((len(arr) / 2.0) / total)
        if np.any(arr < 0) or np.any(arr > 1):
            return DEFAULT_ERROR_RETURN
    dx = 2.0 / len(arr)
    convolution = np.correlate(arr, 1.0 - arr, mode="full") * dx
    c5 = float(np.max(convolution))
    if not np.isfinite(c5):
        return DEFAULT_ERROR_RETURN
    return c5


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    with open(args.input, "r") as handle:
        data = json.load(handle)
    ok, msg = validate_solution(data)
    print(compute_objective_value(data) if ok else f"INVALID: {msg}")
