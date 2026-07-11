import os
import sys

sys.path.append(os.path.dirname(__file__))

from openevolve.modular_utils.evaluation_controller import (
    ObjectiveBasedProblemEvaluator,
    create_evaluation_controller,
    get_current_problem_config,
)
from verify import compute_objective_value, validate_solution


PROBLEM_CONFIG = get_current_problem_config()


def evaluate(program_path, temp_dir=None):
    problem_evaluator = ObjectiveBasedProblemEvaluator(
        validate_func=validate_solution,
        compute_objective_func=compute_objective_value,
        maximize=False,
    )
    evaluation_controller = create_evaluation_controller(problem_evaluator, PROBLEM_CONFIG)
    return evaluation_controller.evaluate(program_path, temp_dir)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = evaluate(sys.argv[1])
        print(f"Score: {result.metrics.get('combined_score')}")
        print(f"Objective: {result.metrics.get('objective_value')}")
    else:
        print("Usage: python evaluator_modular.py <program_path>")
