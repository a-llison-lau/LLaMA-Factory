# srun --mem=48GB -c 4 --gres=gpu:1 --time=2-00:00:00 --qos=long -p a40 -n 1 --pty bash
# !import code; code.interact(local=vars())
import random
from typing import Optional, Tuple, Dict, Any
from torch.utils.data import DataLoader

import json

import torch
import transformers
from transformers import HfArgumentParser, Seq2SeqTrainingArguments, AutoModelForCausalLM
from datasets import load_dataset

from llamafactory.train.callbacks import LogCallback
from llamafactory.train.trainer_utils import get_batch_logps
from llamafactory.train.dpo.trainer import CustomDPOTrainer

from llamafactory.model import load_model, load_tokenizer

from llamafactory.data import PairwiseDataCollatorWithPadding, get_dataset_mod, get_template_and_fix_tokenizer

from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.misc import get_current_device

from llamafactory.hparams.parser import get_train_args
from llamafactory.hparams.data_args import DataArguments
# from ..hparams.evaluation_args import EvaluationArguments
from llamafactory.hparams.finetuning_args import FinetuningArguments
from llamafactory.hparams.generating_args import GeneratingArguments
from llamafactory.hparams.model_args import ModelArguments

_TRAIN_ARGS = [ModelArguments, DataArguments, Seq2SeqTrainingArguments, FinetuningArguments, GeneratingArguments]
_TRAIN_CLS = Tuple[ModelArguments, DataArguments, Seq2SeqTrainingArguments, FinetuningArguments, GeneratingArguments]

# dataset_file = "/h/laualli1/LLaMA-Factory/data/keertana_dpo_1000.json"
YAML_PATH = "/h/laualli1/LLaMA-Factory/examples/train_lora/llama3_lora_dpo.yaml"
ADAPTER_MODEL_SAVEPATH = "/scratch/ssd004/scratch/laualli1/qwen1.5B_instruct_ppt_keertana_1000_cutofflen2048_adapter"
MERGED_MODEL_SAVEPATH = "/scratch/ssd004/scratch/laualli1/qwen1.5B_instruct_ppt_keertana_1000_sysprompt_cutofflen2048_merged"
TEST_MODEL_PATH = "/scratch/ssd004/scratch/laualli1/qwen_sft_test"

### Create eval dataset ###
GENERATE_EVAL_DATASET = False
if GENERATE_EVAL_DATASET:
    DATASET_SAVEPATH = "/h/laualli1/LLaMA-Factory/data/"
    MAX_NUM_TURNS = 8
    NUM_TRIALS = 10
    dataset = load_dataset("keertanavc/imdb_sentiment_grammar_dpo_multipreference")
    eval_dataset = dataset['train']

    organized_data = {}
    for row in eval_dataset:
        human_label = row['human_label']
        
        if human_label not in organized_data:
            organized_data[human_label] = {'prompt': [], 'chosen': [], 'rejected': []}
        
        organized_data[human_label]['prompt'].append(row['prompt'])
        organized_data[human_label]['chosen'].append(row['chosen'])
        organized_data[human_label]['rejected'].append(row['rejected'])

    # Pick one human label from 0 to 4 and one from 5 to 19
    label1 = random.randint(0, 4)
    label2 = random.randint(5, 19)

    # Isolate data for the two selected labels
    data_label1 = organized_data[label1]
    data_label2 = organized_data[label2]

    def create_group_data(data, num_trials, max_num_turns):
        """Create groups of `num_trials` and `max_num_turns` for the selected human label
        """
        groups = []
        
        # Randomly pick `num_trials` groups of `max_num_turns` datapoints
        for _ in range(num_trials):
            group = {
                "prompt": [],
                "chosen": [],
                "rejected": []
            }
            selected_indices = random.sample(range(len(data['chosen'])), max_num_turns)
            
            for idx in selected_indices:
                group["prompt"].append(data['prompt'][idx])
                group["chosen"].append(data['chosen'][idx])
                group["rejected"].append(data['rejected'][idx])

            groups.append(group)

        return groups

    group_data_label1 = create_group_data(data_label1, NUM_TRIALS, MAX_NUM_TURNS)
    group_data_label2 = create_group_data(data_label2, NUM_TRIALS, MAX_NUM_TURNS)

    def format_json_data(group_data, max_num_turns):
        """Format group data into the required JSON structure, excluding the last GPT response from the conversation history."""
        json_data = []

        for turn in range(max_num_turns):
            file_data = []
            for trial in group_data:
                conversation = []

                # Build the conversation up to the current turn (exclude GPT response for the last turn)
                for t in range(turn):
                    # Add the human prompt
                    conversation.append({"from": "human", "value": trial["prompt"][t]})
                    
                    # Add the GPT's chosen and rejected responses for previous turns
                    conversation.append({
                        "from": "gpt",
                        "value": f"<chosen> {trial['chosen'][t]} <rejected> {trial['rejected'][t]}"
                    })

                # Add the human prompt for the current turn
                conversation.append({"from": "human", "value": trial["prompt"][turn]})

                # Add the chosen and rejected responses for the current turn at the top level
                data_entry = {
                    "conversations": conversation,
                    "chosen": {"from": "gpt", "value": trial["chosen"][turn]},
                    "rejected": {"from": "gpt", "value": trial["rejected"][turn]}
                }
                file_data.append(data_entry)
            json_data.append(file_data)

        return json_data

    json_data_label1 = format_json_data(group_data_label1, MAX_NUM_TURNS)
    json_data_label2 = format_json_data(group_data_label2, MAX_NUM_TURNS)

    def save_json_data(json_data, label, num_turns, save_path):
        """Save the data to JSON files
        """
        for turn in range(num_turns):
            filename = save_path + f"train_keertana_dpo_label_{label}_turn_{turn + 1}.json"
            with open(filename, 'w') as f:
                json.dump(json_data[turn], f, indent=2)

    save_json_data(json_data_label1, label1, MAX_NUM_TURNS, DATASET_SAVEPATH)
    save_json_data(json_data_label2, label2, MAX_NUM_TURNS, DATASET_SAVEPATH)

### Functions (modified) from llamafactory ###

def parse_args(parser: "HfArgumentParser", args: Optional[Dict[str, Any]] = None) -> Tuple[Any]:
    return parser.parse_yaml_file(YAML_PATH)

def _parse_train_args(args: Optional[Dict[str, Any]] = None) -> _TRAIN_CLS:
    parser = HfArgumentParser(_TRAIN_ARGS)
    return parse_args(parser, args)

def get_train_args(args: Optional[Dict[str, Any]] = None) -> _TRAIN_CLS:
    model_args, data_args, training_args, finetuning_args, generating_args = _parse_train_args(args)

    print("parsed train args, checking")

    if data_args.neat_packing and not data_args.packing:
        data_args.packing = True

    # Post-process model arguments
    if training_args.bf16 or finetuning_args.pure_bf16:
        model_args.compute_dtype = torch.bfloat16
    elif training_args.fp16:
        model_args.compute_dtype = torch.float16

    model_args.device_map = {"": get_current_device()}
    model_args.model_max_length = data_args.cutoff_len
    model_args.block_diag_attn = data_args.neat_packing
    data_args.packing = data_args.packing if data_args.packing is not None else finetuning_args.stage == "pt"

    transformers.set_seed(training_args.seed)

    return model_args, data_args, training_args, finetuning_args, generating_args

def compute_rm(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    turn_eval_datasets,
    beta
):
    callbacks = []
    callbacks.append(LogCallback())

    # Loads pretrained tokenizer and optionally loads processor
    tokenizer_module = load_tokenizer(model_args)   # safe
    tokenizer = tokenizer_module["tokenizer"]

    # Gets chat template and fixes the tokenizer
    template = get_template_and_fix_tokenizer(tokenizer, data_args) # safe

    # Load the model
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)  # Set is_trainable to False

    # Update training arguments
    training_args.remove_unused_columns = False
    training_args.do_train = False
    training_args.do_eval = True

    # # Reference model (frozen)
    ref_model = model
    ref_model.eval()

    policy_model = AutoModelForCausalLM.from_pretrained(MERGED_MODEL_SAVEPATH)
    policy_model.eval()
    policy_model.to(training_args.device)

    margins = []
    accuracies = []
    for turn_eval_dataset in turn_eval_datasets:

        data_args.train_dataset = None
        data_args.eval_dataset = [turn_eval_dataset]

        dataset_module = get_dataset_mod(template, model_args, data_args, training_args, stage="rm", **tokenizer_module)    # safe
        eval_dataset = dataset_module.get("eval_dataset")

        # Initialize the data collator
        data_collator = PairwiseDataCollatorWithPadding(
            template=template,
            model=ref_model,
            pad_to_multiple_of=8,
            label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
            **tokenizer_module,
        )

        dataloader = DataLoader(
            eval_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=data_collator,
            shuffle=True
        )

        batch_accuracies = []
        batch_margins = []
        # Loop over each datapoint
        for batch in dataloader:
            # batch['input_ids'] shape (2, 136) contains one datapoint all token ids
            # batch['labels'] shape (2, 136) contains only chosen or rejected ids from last turn
            
            batch = {k: v.to(training_args.device) for k, v in batch.items()}
            batch = {k: v.detach().clone() for k, v in batch.items()}

            # Same calculation from concatenated_forward in dpo/trainer.py
            # Compute logits from the policy model
            all_policy_logits = policy_model(
                **batch, return_dict=True, use_cache=False
            ).logits.to(torch.float32) # (2, 136, 151936)

            # Computes logits 
            all_policy_logps, _ = get_batch_logps(logits=all_policy_logits, labels=batch["labels"]) # (2)

            # Compute logits from the reference model
            all_ref_logits = ref_model(
                **batch, return_dict=True, use_cache=False
            ).logits.to(torch.float32)
            all_ref_logps, _ = get_batch_logps(logits=all_ref_logits, labels=batch["labels"])

            # Split logits into chosen and rejected for both models
            batch_size = batch["input_ids"].size(0) // 2
            policy_chosen_logps, policy_rejected_logps = all_policy_logps.split(batch_size, dim=0)
            ref_chosen_logps, ref_rejected_logps = all_ref_logps.split(batch_size, dim=0)

            ##### DPOTrainer ####
            chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
            rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
            margin = (chosen_rewards - rejected_rewards).item()
            ####

            ##### Mine ####
            # Compute rewards for policy model and reference model
            # chosen_rewards = torch.exp((policy_chosen_logps - ref_chosen_logps)) * beta
            # rejected_rewards = torch.exp((policy_rejected_logps - ref_rejected_logps)) * beta
            accuracy = (chosen_rewards) >= (rejected_rewards)
            batch_accuracies.append(accuracy)
            # # Compute the reward margin
            # margin = chosen_rewards.item() - rejected_rewards.item()
            ####
            
            batch_margins.append(margin)
        
        accuracies.append(sum(batch_accuracies)/len(batch_accuracies))
        margins.append(sum(batch_margins)/len(batch_margins))

    return margins, accuracies


### Run ###
model_args, data_args, training_args, finetuning_args, generating_args = get_train_args()
rm, accuracy = compute_rm(model_args, data_args, training_args, finetuning_args, turn_eval_datasets=["train_keertana_dpo_label_4_turn_1_sysprompt", "train_keertana_dpo_label_4_turn_2_sysprompt", "train_keertana_dpo_label_4_turn_3_sysprompt", "train_keertana_dpo_label_4_turn_4_sysprompt"], beta=0.2)

print(f"reward margin: {rm}")
print(f"accuracy: {accuracy}")

