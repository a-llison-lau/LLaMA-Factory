from datasets import load_dataset
import datasets
from huggingface_hub import hf_hub_download
import random
from tqdm import tqdm
import json

NUM_DATAPOINTS_PER_GROUP = 1000
MAX_NUM_TURNS = 4
OUTPUT_FILE = "/h/laualli1/LLaMA-Factory/data/keertana_dpo_standard_unshuffled.json"

# compute canada #
file_path = hf_hub_download(repo_id="keertanavc/imdb_sentiment_grammar_dpo_multipreference", filename="data.arrow")
dataset = datasets.DatasetDict.load_from_disk(file_path)
###

# dataset = load_dataset("keertanavc/imdb_sentiment_grammar_dpo_multipreference")
train_dataset = dataset['train']

# import pdb; pdb.set_trace()

organized_data = {}
for row in train_dataset:
    human_label = row['human_label']
    
    if human_label not in organized_data:
        organized_data[human_label] = {'prompt': [], 'chosen': [], 'rejected': []}
    
    organized_data[human_label]['prompt'].append(row['prompt'])
    organized_data[human_label]['chosen'].append(row['chosen'])
    organized_data[human_label]['rejected'].append(row['rejected'])

# import pdb; pdb.set_trace()

def create_datapoints_standard(organized_data):

    all_data = []
    for human_label, data in organized_data.items():
        prompts = data["prompt"]
        chosens = data["chosen"]
        rejecteds = data["rejected"]

        for i in tqdm(range(len(prompts)), desc=f"Processing human_label {human_label}"):

            datapoint = {
                "conversations": [{"from": "human", "value": prompts[i]}],
                "chosen": {"from": "gpt", "value": f"{chosens[i]}"},
                "rejected": {"from": "gpt", "value": f"{rejecteds[i]}"},
            }

            all_data.append(datapoint)

    return all_data

def create_datapoints(organized_data, num_datapoints_per_group, max_num_turns):
    all_data = []
    for human_label, data in organized_data.items():
        # Process data for this human group
        prompts = data["prompt"]
        chosens = data["chosen"]
        rejecteds = data["rejected"]

        for _ in tqdm(range(num_datapoints_per_group), desc=f"Processing human_label {human_label}"):
            num_turns = random.randint(1, max_num_turns)
            sampled_indices = random.sample(range(len(prompts)), num_turns)

            conversations = []
            for idx in sampled_indices[:-1]:
                conversations.append({"from": "human", "value": prompts[idx]})
                conversations.append({"from": "gpt", "value": f"<chosen> {chosens[idx]} <rejected> {rejecteds[idx]}"})

            # Add the last turn
            last_idx = sampled_indices[-1]
            conversations.append({"from": "human", "value": prompts[last_idx]})

            # Final chosen and rejected values
            chosen_answer = chosens[last_idx]
            rejected_answer = rejecteds[last_idx]

            datapoint = {
                "conversations": conversations,
                "chosen": {"from": "gpt", "value": f"{chosen_answer}"},
                "rejected": {"from": "gpt", "value": f"{rejected_answer}"},
            }

            all_data.append(datapoint)
    return all_data

# Generate the data
# data = create_datapoints(organized_data, NUM_DATAPOINTS_PER_GROUP, MAX_NUM_TURNS)
data = create_datapoints_standard(organized_data)
# random.shuffle(data)

# Save to a JSON file
try:
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Data successfully saved to {OUTPUT_FILE}")
except OSError as e:
    print(f"Error saving file: {e}")

