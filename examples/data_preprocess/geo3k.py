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
"""
Preprocess the Geometry3k dataset to parquet format
"""

import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    r"process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively, and put your final answer in \boxed{}. "
    r"i.e., <think> reasoning process here </think><answer> answering process here \boxed{final answer} </answer>"
)

USER_PROMPT_FORMAT = (
    "Please reason step by step, and put your answer within \\boxed{{final answer}} inside <answer></answer> tags,"
    " i.e., <answer>answering process \\boxed{{final answer}}</answer>. "
    "Users question is: {prompt}"
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/geo3k')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    # data_source = 'hiyouga/geometry3k'
    data_source = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hldy-nlp/FMG/zhuangziyuan/DATA/geo3k/raw"

    dataset = datasets.load_dataset(data_source)

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    # instruction_following = r"Please reason step by step, and put your final answer within \boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            problem = example.pop('problem')
            prompt = USER_PROMPT_FORMAT.format(prompt=problem)
            answer = example.pop('answer')
            images = example.pop('images')

            data = {
                "data_source": "hiyouga/geometry3k",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "images": images,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": problem,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
