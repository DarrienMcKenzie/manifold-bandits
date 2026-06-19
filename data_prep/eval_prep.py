import argparse
import pandas as pd
import numpy as np
from ipdb import set_trace

def add_math_instructions(prompt_list):
    INSTR_START = "Solve the following math problem step by step. The last line of your response should be of the form \\boxed{$Answer} where $Answer is the answer to the problem.\n\n"
    INSTR_END = "\n\nRemember to put your answer inside a \\boxed{} tag."

    if isinstance(prompt_list, (np.ndarray, list)):
        prompt_list = list(prompt_list)
    else:
        # sometimes it could be stored as string in older parquet
        prompt_list = eval(prompt_list)  # only if safe
    
    prompt_list[0]["content"] = INSTR_START + prompt_list[0]["content"] + INSTR_END

    return prompt_list

def add_mc_instructions(prompt_list):
    #OLD (Attempt 1):
    #INSTR_START = ("Answer the following multiple choice question. Think step by step before answering. End your response with a final line exactly"
    #               " in the format: Answer: $LETTER where LETTER is the label of the correct answer and must be one of the option labels given in the question.\n\n")
    #
    #INSTR_END = "\n\n" 

    # Attempt 2:
    """
    INSTR_START = ("Answer the following multiple choice question.\n"
                    "Reason step by step by identifying the key clues, ruling out alternatives, "
                    "and then selecting the best answer.\n"
                    "The last line must be exactly:\n"
                    "Answer: <LETTER>\n\n")
    INSTR_END = "" #Attempt 1: "\n\n"
    """
    # generic; for Qwen3 (non-base)
    INSTR_START = ("Answer the following multiple choice question.\n"
                  "Reason step by step before selecting the best answer.\n"
                   "The last line must be exactly:\n"
                   "Answer: <LETTER>\n\n")
    INSTR_END = ""

    if isinstance(prompt_list, (np.ndarray, list)):
        prompt_list = list(prompt_list)
    else:
        # sometimes it could be stored as string in older parquet
        prompt_list = eval(prompt_list)  # only if safe
    
    prompt_list[0]["content"] = INSTR_START + prompt_list[0]["content"] + INSTR_END

    return prompt_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="all")
    args = parser.parse_args()

   
    if args.tasks == 'math':
        df = pd.read_parquet("data/math_eval.parquet")
        mask = df["data_source"] != "Idavidrein/gpqa"
        df.loc[mask, "prompt"] = df.loc[mask, "prompt"].apply(add_math_instructions)
        df.to_parquet("data/math_eval.parquet", index=False)
    elif args.tasks == 'med':
        eval_df = pd.read_parquet("data/med_eval.parquet")
        eval_df["prompt"] = eval_df["prompt"].apply(add_mc_instructions)
        eval_df.to_parquet("data/med_eval.parquet", index=False)
    elif args.tasks == 'alphamed':
        # no mask
        train_df = pd.read_parquet("data/alphamed-19k.parquet")
        #train_df.loc["prompt"] = train_df.loc["prompt"].apply(add_mc_instructions)
        train_df["prompt"] = train_df["prompt"].apply(add_mc_instructions)
        train_df.to_parquet("data/alphamed-19k.parquet", index=False)

if __name__=="__main__":
    main()