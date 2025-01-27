import json

input_filepath = "/h/laualli1/LLaMA-Factory/data/keertana_dpo_1000.json"
output_filepath = "/h/laualli1/LLaMA-Factory/data/keertana_dpo_1000_sysprompt.json"

with open(input_filepath, "r", encoding="utf-8") as infile:
    data = json.load(infile)

# Modify the last human prompt in each datapoint
for datapoint in data:
    # Ensure there is at least one conversation entry
    if "conversations" in datapoint and datapoint["conversations"]:
        last_entry = datapoint["conversations"][-1]
        if last_entry["from"] == "human":
            last_entry["value"] = "Generate completion: " + last_entry["value"]

with open(output_filepath, "w", encoding="utf-8") as outfile:
    json.dump(data, outfile, indent=4, ensure_ascii=False)

print(f"Modified JSON saved to {output_filepath}")
