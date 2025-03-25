from .mathruler.grader import extract_boxed_content, grade_answer


def compute_score(predict_str: str, ground_truth: str) -> float:
    try:
        answer = extract_boxed_content(predict_str)
        if ground_truth.lower() == str(answer).lower():
            return 1.0
        if grade_answer(answer, ground_truth):
            return 1.0
    except Exception as e:
        print(e)
        return 0.0
    return 0.0