import json
import matplotlib.pyplot as plt
import numpy as np
import os

# 1个GPU
with open('test_affinity_debug/acc_1process_4gpus/1_gpu/all_inference_time.json', 'r') as f1:
    centralized_data = json.load(f1)

# 平均放置 0~15在GPU0
with open('test_affinity_debug/acc_1process_4gpus/average/all_inference_time.json', 'r') as f2:
    vanilla_data = json.load(f2)

# 完全按照亲和性聚类结果放
with open('test_affinity_debug/acc_1process_4gpus/affinity/all_inference_time.json', 'r') as f3:
    affinity_data = json.load(f3)

# 聚类结果调整到一层每个GPU16个
with open('test_affinity_debug/acc_1process_4gpus/balance/all_inference_time.json', 'r') as f4:
    balance_affinity_data = json.load(f4)

# 聚类结果调整到一层每个GPU32个
with open('test_affinity_debug/acc_1process_4gpus/balance_2/all_inference_time.json', 'r') as f5:
    balance_affinity_data_2 = json.load(f5)

fig_dir = f"inference_time_figs/1process_4gpus"
os.makedirs(fig_dir, exist_ok=True)

prompt_sizes = [8, 16, 32, 64, 128, 256, 512, 1024] # 1024

centralized_times = [centralized_data[str(prompt_num)] for prompt_num in prompt_sizes]
vanilla_times = [vanilla_data[str(prompt_num)] for prompt_num in prompt_sizes]
affinity_times = [affinity_data[str(prompt_num)] for prompt_num in prompt_sizes]
balance_affinity_times = [balance_affinity_data[str(prompt_num)] for prompt_num in prompt_sizes]
balance_affinity_times_2 = [balance_affinity_data_2[str(prompt_num)] for prompt_num in prompt_sizes]

# 平均每个 prompt 推理时间
centralized_times_per_prompt = [time / prompt_num for time, prompt_num in zip(centralized_times, prompt_sizes)]
vanilla_times_per_prompt = [time / prompt_num for time, prompt_num in zip(vanilla_times, prompt_sizes)]
affinity_times_per_prompt = [time / prompt_num for time, prompt_num in zip(affinity_times, prompt_sizes)]
balance_affinity_times_per_prompt = [time / prompt_num for time, prompt_num in zip(balance_affinity_times, prompt_sizes)]
balance_affinity_times_2_per_prompt = [time / prompt_num for time, prompt_num in zip(balance_affinity_times_2, prompt_sizes)]

x_positions = np.linspace(0, len(prompt_sizes) - 1, len(prompt_sizes))

# 总时间
total_fig_path = os.path.join(fig_dir, "total_inference_comparison.png")
plt.figure(figsize=(10, 5))
plt.plot(x_positions, centralized_times, marker='o', label='Centralize Placement(1 gpu)')
plt.plot(x_positions, vanilla_times, marker='s', label='Average Placement(4 gpus)')
plt.plot(x_positions, affinity_times, marker='^', label='Affinity Placement(4 gpus)')
plt.plot(x_positions, balance_affinity_times, marker='D', label='Balanced Affinity Placement(4 gpus)')
plt.plot(x_positions, balance_affinity_times_2, marker='o', label='Balanced Affinity Placement(2 gpus)')
plt.xlabel("Num of Prompts")
plt.ylabel("Total Inference Time (s)")
plt.title("Total Inference Time Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xticks(x_positions, labels=[str(x) for x in prompt_sizes], fontsize=12)
plt.savefig(total_fig_path)
plt.close()

# 平均
average_fig_path = os.path.join(fig_dir, "average_inference_comparison.png")
plt.figure(figsize=(10, 5))
plt.plot(x_positions, centralized_times_per_prompt, marker='o', label='Centralize Placement(1 gpu)')
plt.plot(x_positions, vanilla_times_per_prompt, marker='s', label='Average Placement(4 gpus)')
plt.plot(x_positions, affinity_times_per_prompt, marker='^', label='Affinity Placement(4 gpus)')
plt.plot(x_positions, balance_affinity_times_per_prompt, marker='D', label='Balanced Affinity Placement(4 gpus)')
plt.plot(x_positions, balance_affinity_times_2_per_prompt, marker='o', label='Balanced Affinity Placement(2 gpus)')
plt.xlabel("Num of Prompts")
plt.ylabel("Avg Time per Prompt (s)")
plt.title("Average Time per Prompt Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xticks(x_positions, labels=[str(x) for x in prompt_sizes], fontsize=12)
plt.savefig(average_fig_path)
plt.close()
