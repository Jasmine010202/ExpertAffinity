import random
import os

# 加载 prompts
with open("./dataset/sonnet/sonnet.txt", "r") as file:
    prompts = [line.strip() for line in file.readlines() if line.strip()]

output_dir = "./dataset/sonnet/used_for_occult"
os.makedirs(output_dir, exist_ok=True)

prompt_sizes = [512] # 8, 16, 32, 64, 128, 256, 512, 1024

for size in prompt_sizes:
    print(f"\nPrompt size: {size}")
    # sampled_prompts = random.sample(prompts, size)  # 随机抽取
    sampled_prompts = prompts[:size]    # 前x个

    sub_dataset_path = os.path.join(output_dir, f"sonnet_{size}.txt")
    with open(sub_dataset_path, "w", encoding="utf-8") as output_file:
        for prompt in sampled_prompts:
            output_file.write(f"{prompt}\n")