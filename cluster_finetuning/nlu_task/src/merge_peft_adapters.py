from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

import os
import argparse

blob_base_model_path = "path/to/foundation/model/on/azure/storage/blob"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datastore_path", type=str)
    parser.add_argument("--base_model_name_or_path", type=str, default=blob_base_model_path)
    parser.add_argument("--peft_model_path", type=str)
    parser.add_argument("--merged_model_output_dir", type=str)

    return parser.parse_args()


def main():
    args = get_args()

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        os.path.join(args.datastore_path, args.base_model_name_or_path), 
        return_dict=True, 
        torch_dtype=torch.float16
    )

    print("Loading peft model...")
    model = PeftModel.from_pretrained(base_model, os.path.join(args.datastore_path, args.peft_model_path))

    print("Merging...")
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.datastore_path, args.base_model_name_or_path))

    model.save_pretrained(os.path.join(args.datastore_path, args.merged_model_output_dir))
    tokenizer.save_pretrained(os.path.join(args.datastore_path, args.merged_model_output_dir))
    print(f"Merged model saved.")


if __name__ == "__main__":
    main()
