from datasets import load_dataset
from ipdb import set_trace

def main():
    ds = load_dataset("parquet", data_files='data/dapo-math-17k.parquet', split='train').rename_column("prompt", "raw_prompt").rename_column('source_prompt', 'prompt').remove_columns("raw_prompt")
    ds.to_parquet('data/dapo-math-17k.parquet')
    #print("[DEBUG] File overwritten!")

if __name__=='__main__':
    main()