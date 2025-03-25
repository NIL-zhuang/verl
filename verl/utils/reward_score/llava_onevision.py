from .mathruler.grader import extract_boxed_content
from math_verify import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig, parse, verify


def compute_score(predict_str: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict_str)
    if ground_truth.lower() == str(answer).lower():
        return 1.0

    # parse gold answer
    try:
        gold_parsed = parse(ground_truth)
        if len(gold_parsed) == 0:
            return 0.0
    except Exception:
        return 0.0

    # parse model prediction
    try:
        answer_parsed = parse(
            predict_str, extraction_config=[StringExtractionConfig(), LatexExtractionConfig(), ExprExtractionConfig()]
        )
        reward = float(verify(answer_parsed, gold_parsed))
    except Exception:
        reward = 0.0

    return reward