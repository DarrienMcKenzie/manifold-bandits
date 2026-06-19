# bmc_reward.py
import re
from typing import Iterable
from verl.utils.reward_score import default_compute_score
from process_safe_math_verify import process_safe_verify

# --- Configs ---
VERIFY_TIMEOUT_SECONDS = 5.0
VERIFY_CONFIG_MODES = ["expr", "latex"]

# --- Math Datasets ---
MATH_EVAL_DATASETS = [
    "Maxwell-Jia/AIME_2024",
    "opencompass/cnmo2024_zh",
    "yentinglin/aime_2025",
    "HuggingFaceH4/MATH-500",
    "AI-MO/aimo-validation-amc",
]

MED_EVAL_DATASETS = [
    "che111/AlphaMed19K",
    "GBaker/MedQA-USMLE-4-options",
    "CAIR-HKISI/medmcqa_test",
    "TsinghuaC3I/MedXpertQA"
]

GPQA_DATASETS = ["Idavidrein/gpqa"]

ALL_EVAL_DATASETS = MATH_EVAL_DATASETS + GPQA_DATASETS + MED_EVAL_DATASETS

def extract_boxed(expr: str) -> str:
    """
    Extracts content from the first \boxed{...} in expr.
    Returns empty string if none found.
    """
    match = re.search(r"\\boxed\{(.+?)\}", expr)
    return match.group(1) if match else ""

def extract_mc_answer(
    solution_str: str,
    patterns: Iterable[str] | None = None,
    use_last_match: bool = True,
) -> str | None:
    if solution_str is None:
        return None

    # Just match any uppercase letter
    letter_class = "A-Z"

    if patterns is None:
        patterns = [
            r"(?i)\bfinal\s+answer\b[ \t]*[:=]?[ \t]*\$?\(?\[?\**([A-Z])\**\]?\)?\$?",
            r"(?i)\banswer\b[ \t]*[:=]?[ \t]*\$?\(?\[?\**([A-Z])\**\]?\)?\$?",
            r"(?i)\bthe\s+answer\s+is\b[ \t]*\$?\(?\[?\**([A-Z])\**\]?\)?\$?",
            r"(?i)\bcorrect\s+answer\b[ \t]*[:=]?[ \t]*\$?\(?\[?\**([A-Z])\**\]?\)?\$?",
            r"(?i)\\boxed\{([A-Z])\}",
            r"(?i)\(\s*([A-Z])\s*\)",
            r"(?i)\[\s*([A-Z])\s*\]",
            r"(?i)\b([A-Z])\b[\.\!\s]*$",
        ]

    for pattern in patterns:
        matches = re.findall(pattern, solution_str)
        if matches:
            ans = matches[-1] if use_last_match else matches[0]
            return ans.upper()

    return None

# ------------------------------
# Math-Verify Scoring
# ------------------------------
def compute_score_math(model_output: str, ground_truth: str) -> float:
    """
    Math-Verify scoring using only the boxed answer from the model.
    Compares against the unboxed gold answer.
    """
    # Extract boxed term from model output
    answer = extract_boxed(model_output)
    if not answer:
        return 0.0

    # Process-isolated Math-Verify
    score = process_safe_verify(
        gold_text=str(ground_truth),  # unboxed
        pred_text=answer,         # extracted boxed content
        modes=VERIFY_CONFIG_MODES,
        timeout_seconds=VERIFY_TIMEOUT_SECONDS
    )

    return float(score) if score is not None else 0.0


def compute_score_mc(solution_str: str, ground_truth: str, punishment: float = 0.0, patterns: Iterable[str] | None = None, use_last_match: bool = True) -> float:
    extracted_answer = extract_mc_answer(
        solution_str=solution_str,
        patterns=patterns,
        use_last_match=use_last_match,
    )
    return 1.0 if extracted_answer == ground_truth.upper().strip() else punishment

# ------------------------------
# GPQA Multiple Choice Scoring
# ------------------------------
def compute_score_gpqa(solution_str: str, ground_truth: str) -> float: # keeping original for reproducability, other datasets use generic MC compute
    """
    Extracts answer (A-D) from model output and scores against ground truth.
    """
    ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"
    match = re.search(ANSWER_PATTERN_MULTICHOICE, solution_str)
    extracted_answer = match.group(1) if match else None
    return 1.0 if extracted_answer == ground_truth else 0.0

# ------------------------------
# VERL Reward Function
# ------------------------------
def reward_func(data_source, solution_str, ground_truth, extra_info=None) -> float:
    """
    Custom reward function for VERL.
    Uses:
      - Math-Verify for supported math datasets
      - GPQA scoring for GPQA datasets
      - Default compute score for all other datasets
    """
    if data_source in ALL_EVAL_DATASETS:
        if data_source in MATH_EVAL_DATASETS:
            return compute_score_math(solution_str, ground_truth)
        elif data_source in GPQA_DATASETS:
            return compute_score_gpqa(solution_str, ground_truth)
        elif data_source in MED_EVAL_DATASETS:
            if data_source == "che111/AlphaMed19K":
                return compute_score_mc(solution_str, ground_truth, punishment=-1.0)
            else:
                return compute_score_mc(solution_str, ground_truth)
        else:
            # For any other known evaluation dataset, fallback to default
            return default_compute_score(data_source, solution_str, ground_truth, extra_info)
    else:
        # Training / unknown datasets
        return default_compute_score(data_source, solution_str, ground_truth, extra_info)
