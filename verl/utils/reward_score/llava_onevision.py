from .mathruler.grader import extract_boxed_content


def compute_score(predict_str: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict_str)
    return 1.0 if ground_truth in answer else 0.0
