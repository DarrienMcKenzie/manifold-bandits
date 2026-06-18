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
Preprocess the dataset to parquet format
"""

import argparse
import os
from functools import partial

from datasets import concatenate_datasets, load_dataset

from verl.utils.hdfs_io import copy, makedirs


def example_map_fn(example, idx, process_fn, data_source, split):
    question, solution = process_fn(example)
    data = {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": question}],
        "reward_model": {"style": "rule", "ground_truth": solution},
        "extra_info": {"split": split, "index": idx},
    }
    return data


def build_amc2023_dataset():
    def process_amc2023(example):
        return example["problem"], str(example["answer"])

    data_source = "AI-MO/aimo-validation-amc"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = load_dataset(data_source, split="train")
    map_fn = partial(
        example_map_fn, process_fn=process_amc2023, data_source=data_source, split="test"
    )
    dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
    return dataset


def build_math500_dataset():
    def process_math500(example):
        return example["problem"], str(example["answer"])

    data_source = "HuggingFaceH4/MATH-500"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = load_dataset(data_source, split="test")
    map_fn = partial(
        example_map_fn, process_fn=process_math500, data_source=data_source, split="test"
    )
    dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
    return dataset


def build_aime2025_dataset():
    def process_aime2025(example):
        return example["problem"], str(example["answer"])

    data_source = "yentinglin/aime_2025"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = load_dataset(data_source, split="train")
    map_fn = partial(
        example_map_fn, process_fn=process_aime2025, data_source=data_source, split="test"
    )
    dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
    return dataset


def build_aime2024_dataset():
    def process_aime2024(example):
        return example["Problem"], str(example["Answer"])

    data_source = "Maxwell-Jia/AIME_2024"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = load_dataset(data_source, split="train")
    map_fn = partial(
        example_map_fn, process_fn=process_aime2024, data_source=data_source, split="test"
    )
    dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
    return dataset


def build_gpqa_dimond_dataset():
    import random

    GPQA_QUERY_TEMPLATE = (
        "Answer the following multiple choice question. The last line of your response should be of the following "
        "format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before "
        "answering.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
    )

    def process_gpqa_diamond(example):
        choices = [example["Incorrect Answer 1"], example["Incorrect Answer 2"], example["Incorrect Answer 3"]]
        random.shuffle(choices)
        gold_index = random.randint(0, 3)
        choices.insert(gold_index, example["Correct Answer"])
        query_prompt = GPQA_QUERY_TEMPLATE.format(
            A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=example["Question"]
        )
        gold_choice = "ABCD"[gold_index]
        return query_prompt, gold_choice

    data_source = "Idavidrein/gpqa"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)

    dataset = load_dataset(data_source, "gpqa_diamond", split="train")
    map_fn = partial(
        example_map_fn, process_fn=process_gpqa_diamond, data_source=data_source, split="test"
    )
    dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
    return dataset


def build_cnmo2024_dataset():
    def process_cnmo2024(example):
        return example["question"], example["answer"]

    data_source = "opencompass/LiveMathBench"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)

    """
    dataset_en = load_dataset(data_source, "v202412_CNMO_en", split="test")
    map_fn_en = partial(
        example_map_fn, process_fn=process_cnmo2024, data_source="opencompass/cnmo2024_en", ability="Math", split="test"
    )
    dataset_en = dataset_en.map(map_fn_en, with_indices=True, remove_columns=dataset_en.column_names)
    """

    dataset_cnmo = load_dataset(data_source, "v202412_CNMO_cn", split="test")
    map_fn_zh = partial(
        example_map_fn, process_fn=process_cnmo2024, data_source="opencompass/cnmo2024_zh", split="test"
    )
    dataset_cnmo = dataset_cnmo.map(map_fn_zh, with_indices=True, remove_columns=dataset_cnmo.column_names)

    dataset_ccee = load_dataset(data_source, "v202412_CCEE_cn", split="test")
    map_fn_zh = partial(
        example_map_fn, process_fn=process_cnmo2024, data_source="opencompass/cnmo2024_zh", split="test"
    )
    dataset_ccee = dataset_ccee.map(map_fn_zh, with_indices=True, remove_columns=dataset_ccee.column_names)

    dataset = concatenate_datasets([dataset_cnmo, dataset_ccee])
    return dataset


def build_alphamed_dataset():
    def process_alphamed(example):
        return example["prompt"], str(example["answer"])

    data_source = "che111/AlphaMed19K"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = load_dataset(data_source, split="train")
    dataset = dataset.rename_column("question", "prompt")

    map_fn = partial(
        example_map_fn, process_fn=process_alphamed, data_source=data_source, split="train"
    )
    dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
    return dataset

def build_medqa_dataset():
    def process_medqa(example):
        question = example["question"]
        options = example["options"]
        option_text = "\n".join(
            f"{key}. {value}" for key, value in sorted(options.items())
        )
        full_question = f"{question}\n\n{option_text}"
        return full_question, str(example["answer_idx"])

    data_source = "GBaker/MedQA-USMLE-4-options"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = load_dataset(data_source, split="test")
    map_fn = partial(
        example_map_fn, process_fn=process_medqa, data_source=data_source, split="test"
    )
    dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
    return dataset

def build_medmcqa_dataset():
    def process_medmcqa(example):
        # removing redundant instruction:
        cleaned_question = example["problem"].replace("Please answer with the correct option letter only", "").strip()
        return cleaned_question, str(example["answer"])

    data_source = "CAIR-HKISI/medmcqa_test"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = load_dataset(data_source, split="train")
    map_fn = partial(
        example_map_fn, process_fn=process_medmcqa, data_source=data_source, split="train"
    )
    dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
    return dataset

def build_medxpertqa_dataset():
    def process_medxpertqa(example):
        return example["question"], str(example["label"])

    data_source = "TsinghuaC3I/MedXpertQA"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = load_dataset(data_source, "Text", split="test")
    map_fn = partial(
        example_map_fn, process_fn=process_medxpertqa, data_source=data_source, split="test"
    )
    dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
    return dataset


TASKMAP_MATH = {
    "aime2024": build_aime2024_dataset,
    "gpqa_diamond": build_gpqa_dimond_dataset,
    "cnmo2024": build_cnmo2024_dataset,
    "math500": build_math500_dataset,
    "aime2025": build_aime2025_dataset,
    "amc2023": build_amc2023_dataset,
}

TASKMAP_MED = {"medqa": build_medqa_dataset,
               "medmcqa": build_medmcqa_dataset,
               "medxpertqa": build_medxpertqa_dataset}

TASKMAP_TRAIN = {"alphamed" : build_alphamed_dataset}

TASKMAP_ALL = {
    **TASKMAP_MATH,
    **TASKMAP_MED,
    **TASKMAP_TRAIN
}

SUPPORTED_TASKS = set(TASKMAP_ALL)


def resolve_tasks(task_arg: str) -> list[str]:
    task_arg = task_arg.strip().lower()

    if task_arg == "all":
        return list(TASKMAP_ALL)
    if task_arg == "math":
        return list(TASKMAP_MATH)
    if task_arg == "med":
        return list(TASKMAP_MED)

    tasks = [task.strip().lower() for task in task_arg.split(",") if task.strip()]
    unsupported = [task for task in tasks if task not in SUPPORTED_TASKS]
    if unsupported:
        raise NotImplementedError(
            f"Unsupported task(s): {unsupported}. Supported tasks are: {sorted(SUPPORTED_TASKS)}"
        )

    return tasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--filename", default="eval.parquet")
    args = parser.parse_args()

    tasks = resolve_tasks(args.tasks)

    datasets = [TASKMAP_ALL[task]() for task in tasks]
    test_dataset = concatenate_datasets(datasets)

    os.makedirs(args.local_dir, exist_ok=True)
    output_path = os.path.join(args.local_dir, args.filename)
    test_dataset.to_parquet(output_path)

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=args.local_dir, dst=args.hdfs_dir)