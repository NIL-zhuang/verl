from typing import Optional, Dict
import re

pattern_without_boxed = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
pattern_with_boxed = r"<think>.*?</think>\s*<answer>.*?\boxed{.*?}.*?</answer>"


def compute_score(data_source: str, solution_str: str, extra_info: Optional[Dict] = None) -> float:
    """
    Reward function that checks if the reasoning process is enclosed within
    <think> and </think> tags
    while the final answer is enclosed within <answer> and </answer> tags.
    """
    if re.match(pattern_with_boxed, solution_str, re.DOTALL | re.MULTILINE):
        return 1.0
    elif re.match(pattern_without_boxed, solution_str, re.DOTALL | re.MULTILINE):
        return 0.5
    else:
        return 0.0
