from typing import Optional, Dict
import re

pattern = r"^<think>.*?</think>.*?\boxed{.*?}.*?$"


def compute_score(data_source: str, solution_str: str, extra_info: Optional[Dict] = None) -> float:
    """
    Reward function that checks if the reasoning process is enclosed within
    <think> and </think> tags
    while the final answer is enclosed within <answer> and </answer> tags.
    """
    if re.match(pattern, solution_str, re.DOTALL | re.MULTILINE):
        return 1.0
    else:
        return 0.0
