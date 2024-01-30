import argparse
import os
import json
import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
import utils
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)
        
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"


MISTRAL_PROMPT_DICT = {
    "prompt_input": ("[INST] {instruction} Input: {input} [/INST]"),
    "prompt_no_input": ("[INST] {instruction} [/INST]"),
}

PROMPT_DICT = {
    "prompt_input": ("{instruction}\n\n### Input:{input}\n\n### Response:"),
    "prompt_no_input": ("{instruction}\n\n### Response:"),
}


blob_base_model_path = "path/to/foundation/model/on/azure/storage/blob"

# change the `wandb_api_key` here
wandb_api_key = "xxxxx"
os.environ["WANDB_API_KEY"] = wandb_api_key
os.environ["TOKENIZERS_PARALLELISM"] = "true"


class SavePeftModelCallback(TrainerCallback):
    """
    Trainer callback function for saving model and LoRA adapter parameters at each 'save_step'.
    """

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Checkpoint folder path should match trainer's default saving path.
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        try:
            kwargs["model"].save_pretrained(checkpoint_folder)
            print("Full model dict saved.")
            # When using DeepSpeed ZeRO stage 3, the model will not save the complete LoRA adapter due to parameter partition.
            # To avoid the incomplete saving, we first collect the full parameters from ZeRO checkpoint, extract those belonging
            # to LoRA, and finally save the full adpater.
            state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_folder)
            peft_state_dict = get_peft_model_state_dict(kwargs["model"], state_dict=state_dict)
            torch.save(peft_state_dict, os.path.join(checkpoint_folder, "adapter_model.bin"))
            print("Lora adapter saved.")
        except Exception as e:
            print(f"Error when trying to save or merge model: {e}")

        return control



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--datastore_path", type=str)
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--blob_data_path", type=str)
    parser.add_argument("--complex_data", type=str, default=None)

    parser.add_argument("--model_max_length", type=int, default=2048)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--evaluation_strategy", type=str, default="no")
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=30)
    parser.add_argument("--logging_steps", default=1, type=int)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--run_name", type=str, default="mistral-finetuned")
    parser.add_argument("--deepspeed", type=str, default="deepspeed_config.json")

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = MISTRAL_PROMPT_DICT["prompt_input"], MISTRAL_PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


class SupervisedComplexDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedComplexDataset, self).__init__()
        logging.warning("Loading data...")
        # list_data_dict = utils.jload(data_path)

        list_data_dict = []

        with open(data_path) as f:
            for line in f:
                cur_obj = json.loads(line)
                cur_obj = json.loads(cur_obj)
                list_data_dict.append(cur_obj)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = MISTRAL_PROMPT_DICT["prompt_input"], MISTRAL_PROMPT_DICT["prompt_no_input"]
        sources = [(example["instruction"].strip() + "\n\n") for example in list_data_dict]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if args.complex_data == "complex":
        train_dataset = SupervisedComplexDataset(tokenizer=tokenizer, data_path=os.path.join(args.datastore_path, args.blob_data_path))
    else:
        train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=os.path.join(args.datastore_path, args.blob_data_path))
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    args = get_args()

    training_args = TrainingArguments(
        output_dir=os.path.join(args.datastore_path, args.checkpoint_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.fp16,
        report_to=args.report_to,
        run_name=args.run_name,
        deepspeed=args.deepspeed,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(os.path.join(args.datastore_path, blob_base_model_path), use_cache=False)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )

    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        os.path.join(args.datastore_path, blob_base_model_path),
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        }
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, args=args)
    # Tell Trainer not to attempt DataParallel
    model.is_parallelizable = True
    model.model_parallel = True

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[SavePeftModelCallback],
        **data_module,
    )

    trainer.train()
    # trainer.save_model(os.path.join(training_args.output_dir, "final_checkpoint"))


if __name__ == "__main__":
    train()
