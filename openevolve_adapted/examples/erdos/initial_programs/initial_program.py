# EVOLVE-BLOCK-START
import time

import numpy as np


def evaluate_erdos(sequence: list[float]) -> float:
    if not isinstance(sequence, list) or not sequence:
        return np.inf
    arr = np.asarray([float(x) for x in sequence], dtype=float)
    if not np.all(np.isfinite(arr)):
        return np.inf
    if np.any(arr < 0) or np.any(arr > 1):
        return np.inf
    total = float(np.sum(arr))
    if abs(total - len(arr) / 2.0) > 1e-6:
        arr = arr * ((len(arr) / 2.0) / total)
        if np.any(arr < 0) or np.any(arr > 1):
            return np.inf
    dx = 2.0 / len(arr)
    convolution = np.correlate(arr, 1.0 - arr, mode="full") * dx
    return float(np.max(convolution))


def propose_candidate(seed=42, budget_s=60, **kwargs):
    np.random.seed(seed)
    deadline = time.time() + max(1, budget_s - 5)
    n = 256
    best = [0.5] * n
    best_score = evaluate_erdos(best)
    step = 0.05

    while time.time() < deadline:
        candidate = np.asarray(best, dtype=float)
        idx = np.random.randint(0, n, size=max(1, n // 12))
        candidate[idx] = np.clip(candidate[idx] + step * np.random.randn(len(idx)), 0.0, 1.0)
        total = float(np.sum(candidate))
        if total > 0:
            candidate = candidate * ((n / 2.0) / total)
        candidate_list = candidate.tolist()
        score = evaluate_erdos(candidate_list)
        if score < best_score:
            best_score = score
            best = candidate_list
        else:
            step = max(0.001, step * 0.995)
    return best
# EVOLVE-BLOCK-END


def run_erdos():
    sequence = propose_candidate(seed=42, budget_s=MAX_TIME)
    score = evaluate_erdos(sequence)
    current_solution = {"data": sequence}
    save_search_results(
        best_perfect_solution=None,
        current_solution=current_solution,
        target_value=TARGET_C5,
        objective_value=score,
    )
    return sequence, score


if __name__ == "__main__":
    from openevolve.modular_utils.evaluation_controller import get_current_problem_config
    from openevolve.modular_utils.file_io_controller import save_search_results

    PROBLEM_CONFIG = get_current_problem_config()
    TARGET_C5 = PROBLEM_CONFIG["core_parameters"]["target_value"]
    PROBLEM_TYPE = PROBLEM_CONFIG["problem_type"]
    MAX_TIME = PROBLEM_CONFIG["time_limit"] * 0.9
    sequence, score = run_erdos()
    print(f"Generated {PROBLEM_TYPE} sequence")
    print(f"Erdos C5: {score:.8f}")
    print(f"Target: {TARGET_C5}")
    print(f"Length: {len(sequence)}")
