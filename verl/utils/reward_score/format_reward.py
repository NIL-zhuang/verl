from typing import Optional, Dict
import re

pattern = r"^<think>.*?</think>.*?\\boxed{.*?}.*?$"


def compute_score(data_source: str, solution_str: str, extra_info: Optional[Dict] = None) -> float:
    """
    Reward function that checks if the reasoning process is enclosed within <think> and </think> tags
    And check if the answer is embraced in \\boxed{}
    """
    p = re.compile(pattern, re.DOTALL | re.MULTILINE)
    format_match = re.fullmatch(p, solution_str)
    return 1.0 if format_match else 0.0


AHA_MOMENT_WORDS = ["rethink", "reconsider", "wait", "re-evaluate", "however"]


def compute_aha_moment(solution_str: str, words: list[str] = AHA_MOMENT_WORDS):
    solution = solution_str.lower()
    total_count = sum(solution.count(word) for word in words)
    return total_count