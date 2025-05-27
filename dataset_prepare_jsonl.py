import json
import random
import os


with open("./dataset/math/GSM8K/test.jsonl", "r", encoding="utf-8") as file:
    prompts = [json.loads(line)["question"] for line in file if line.strip()]

output_dir = "./dataset/math"
os.makedirs(output_dir, exist_ok=True)

prompt_sizes = [8, 16, 32, 64, 128, 256, 512, 1024]

for size in prompt_sizes:
    if size > len(prompts):
        print(f"跳过 size={size}，因为样本数量不足（共 {len(prompts)} 条）")
        continue

    print(f"\nPrompt size: {size}")
    sampled_prompts = random.sample(prompts, size)

    sub_dataset_path = os.path.join(output_dir, f"GSM8K_test_{size}.jsonl")
    with open(sub_dataset_path, "w", encoding="utf-8") as output_file:
        for prompt in sampled_prompts:
            json.dump({"question": prompt}, output_file)
            output_file.write("\n")
