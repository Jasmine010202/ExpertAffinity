import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import time
import torch
import random
import json
from transformers import AutoTokenizer
# from src.modeling_olmoe_parallel import OlmoeForCausalLM # 平均放置
from src.modeling_olmoe_placement import OlmoeForCausalLM # 按层内亲和性放置
# from src.modeling_olmoe import OlmoeForCausalLM
from accelerate import Accelerator

accelerator = Accelerator()
device = accelerator.device

print("🚀 当前进程信息：")
print(f"  ↪ accelerator.process_index = {accelerator.process_index}")
print(f"  ↪ device = {accelerator.device}")
print(f"  ↪ local_process_index = {accelerator.local_process_index}")
print(f"  ↪ total number of processes (GPUs) = {accelerator.num_processes}")
print(f"  ↪ torch.cuda.device_count() = {torch.cuda.device_count()}")

tokenizer = AutoTokenizer.from_pretrained("./models/OLMoE-1B-7B-0924")
tokenizer.padding_side = "left"

# average
#model = OlmoeForCausalLM.from_pretrained("./models/OLMoE-1B-7B-0924", torch_dtype=torch.float16).to(device)

# affinity
model = OlmoeForCausalLM.from_pretrained("./models/OLMoE-1B-7B-0924", torch_dtype=torch.float16)
#model = accelerator.prepare(model)

# accelerate 包装模型（分发到多卡）
#model = accelerator.prepare(model)
model.eval()

# 加载 prompts
with open("./dataset/sonnet.txt", "r") as file:
    prompts = [line.strip() for line in file.readlines() if line.strip()]

prompt_sizes = [8, 16, 32, 64, 128, 256, 512] # 8, 16, 32, 64, 128, 256, 512, 1024
 
batch_size = 8

output_dir = "./test_acc_output_affinity_balance/OLMoE/sonnet"
os.makedirs(output_dir, exist_ok=True)

all_inference_time = {}

for size in prompt_sizes:
    print(f"\nPrompt size: {size}")
    sampled_prompts = random.sample(prompts, size)

    start_time = time.time()
    decoded_outputs = []

    for i in range(0, size, batch_size):
        batch_prompts = sampled_prompts[i:i + batch_size]

        # average
        # inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)

        #affinity
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        inputs = accelerator.prepare(inputs) 

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_outputs.extend(decoded)

    inference_time = time.time() - start_time
    all_inference_time[str(size)] = round(inference_time, 4)

    print(f"Completed size {size} in {inference_time:.4f} seconds")

    # 写入输出
    output_file_path = os.path.join(output_dir, f"output_{size}.txt")
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        for prompt, result in zip(sampled_prompts, decoded_outputs):
            output_file.write(f"Prompt: {prompt}\nGenerated: {result}\n\n")

    print(f"End of inference. ({size} prompts)")

# 保存时间记录
timing_file_path = os.path.join(output_dir, "all_inference_time.json")
with open(timing_file_path, "w") as f:
    json.dump(all_inference_time, f, indent=4)

print(f"All inference time saved to : {timing_file_path}")
