# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# from . import gsm8k, math, prime_math, prime_code
from typing import Callable, Optional, Dict


def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if data_source == 'openai/gsm8k':
        from . import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ['lighteval/MATH', 'DigitalLearningGmbH/MATH-lighteval']:
        from . import math
        res = math.compute_score(solution_str, ground_truth)
    elif data_source in [
            'numina_aops_forum', 'numina_synthetic_math', 'numina_amc_aime', 'numina_synthetic_amc', 'numina_cn_k12',
            'numina_olympiads'
    ]:
        from . import prime_math
        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ['codecontests', 'apps', 'codeforces', 'taco']:
        from . import prime_code
        res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ['hiyouga/geometry3k']:
        from . import geo3k
        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in ["AI4Math/MathVerse", "MathLLMs/MathVision", "AI4Math/MathVista", "lmms-lab/LLaVA-OneVision"]:
        from . import llava_onevision

        res = llava_onevision.compute_score(solution_str, ground_truth)
    else:
        raise NotImplementedError

    if isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])

class RewardFunctionWrapper:
    def __init__(self, compute_score: Optional[Dict[str, Callable] | Callable] = None):
        from . import format_reward

        if compute_score is None:
            self.compute_score = _default_compute_score
        self.format_score = format_reward.compute_score
        self.compute_aha_moment = format_reward.compute_aha_moment
        self.format_score_gamma = 0.1

    def __call__(self, data_source, solution_str, ground_truth, extra_info: dict = None):
        score_reward = self.compute_score(data_source, solution_str, ground_truth, extra_info)
        format_reward = self.format_score(data_source, solution_str, extra_info)
        aha_count = self.compute_aha_moment(solution_str=solution_str)

        reward_metric = {
            "reward/correctness": score_reward,
            "reward/format": format_reward,
            "reward/aha_count": aha_count,
        }
        score = score_reward + self.format_score_gamma * format_reward
        return score, reward_metric
