import json
import matplotlib.pyplot as plt
import numpy as np
import os

# === 加载数据 ===

with open('test_1_gpu/OLMoE/sonnet/all_inference_time.json', 'r') as f1:
    centralized_data = json.load(f1)

with open('test_acc_output_average/OLMoE/sonnet/all_inference_time.json', 'r') as f2:
    vanilla_data = json.load(f2)

with open('test_acc_output_affinity/OLMoE/sonnet/all_inference_time.json', 'r') as f3:
    affinity_data = json.load(f3)

with open('test_acc_output_affinity_balance/OLMoE/sonnet/temp_all_inference_time.json', 'r') as f4:
    balance_affinity_data = json.load(f4)

# with open('test_acc_output_affinity_balance/2/OLMoE/sonnet/temp_all_inference_time.json', 'r') as f5:
#     balance_affinity_data_2 = json.load(f5)

# === 准备绘图 ===

fig_dir = f"inference_time_figs/400"
os.makedirs(fig_dir, exist_ok=True)

prompt_sizes = [8, 16, 32, 64, 128, 256, 512]

def get_time_list(data):
    return [data.get(str(prompt_num), 0) for prompt_num in prompt_sizes]

def truncate_data(data_list, size_list):
    last_valid = 0
    for i in reversed(range(len(data_list))):
        if data_list[i] != 0:
            last_valid = i + 1
            break
    return data_list[:last_valid], size_list[:last_valid]

# 原始数据（总时间）
centralized_times = get_time_list(centralized_data)
vanilla_times = get_time_list(vanilla_data)
affinity_times = get_time_list(affinity_data)
balance_affinity_times = get_time_list(balance_affinity_data)
#balance_affinity_times_2 = get_time_list(balance_affinity_data_2)

# 平均每 prompt 推理时间
centralized_times_per_prompt = [t / p for t, p in zip(centralized_times, prompt_sizes)]
vanilla_times_per_prompt = [t / p for t, p in zip(vanilla_times, prompt_sizes)]
affinity_times_per_prompt = [t / p for t, p in zip(affinity_times, prompt_sizes)]
balance_affinity_times_per_prompt = [t / p for t, p in zip(balance_affinity_times, prompt_sizes)]
#balance_affinity_times_2_per_prompt = [t / p for t, p in zip(balance_affinity_times_2, prompt_sizes)]

x_positions = np.linspace(0, len(prompt_sizes) - 1, len(prompt_sizes))

# === 总推理时间图 ===

total_fig_path = os.path.join(fig_dir, "total_inference_comparison.png")
plt.figure(figsize=(10, 5))

plt.plot(x_positions, centralized_times, marker='o', label='Centralize Placement (1 GPU)')
plt.plot(x_positions, vanilla_times, marker='s', label='Average Placement (4 GPUs)')
plt.plot(x_positions, affinity_times, marker='^', label='Affinity Placement (4 GPUs)')

bal_trim, ps_trim = truncate_data(balance_affinity_times, prompt_sizes)
x_trim = np.linspace(0, len(ps_trim) - 1, len(ps_trim))
plt.plot(x_trim, bal_trim, marker='D', label='Balanced Affinity (4 GPUs)')

# #bal2_trim, ps2_trim = truncate_data(balance_affinity_times_2, prompt_sizes)
# x2_trim = np.linspace(0, len(ps2_trim) - 1, len(ps2_trim))
# plt.plot(x2_trim, bal2_trim, marker='*', label='Balanced Affinity (2 GPUs)')

plt.xlabel("Num of Prompts")
plt.ylabel("Total Inference Time (s)")
plt.title("Total Inference Time Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xticks(np.linspace(0, len(prompt_sizes) - 1, len(prompt_sizes)), labels=[str(x) for x in prompt_sizes], fontsize=12)
plt.savefig(total_fig_path)
plt.close()

# === 平均推理时间图 ===

average_fig_path = os.path.join(fig_dir, "average_inference_comparison.png")
plt.figure(figsize=(10, 5))

plt.plot(x_positions, centralized_times_per_prompt, marker='o', label='Centralize Placement (1 GPU)')
plt.plot(x_positions, vanilla_times_per_prompt, marker='s', label='Average Placement (4 GPUs)')
plt.plot(x_positions, affinity_times_per_prompt, marker='^', label='Affinity Placement (4 GPUs)')

bal_avg_trim, _ = truncate_data(balance_affinity_times_per_prompt, prompt_sizes)
plt.plot(x_trim, bal_avg_trim, marker='D', label='Balanced Affinity (4 GPUs)')

# bal2_avg_trim, _ = truncate_data(balance_affinity_times_2_per_prompt, prompt_sizes)
# plt.plot(x2_trim, bal2_avg_trim, marker='*', label='Balanced Affinity (2 GPUs)')

plt.xlabel("Num of Prompts")
plt.ylabel("Avg Time per Prompt (s)")
plt.title("Average Inference Time Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xticks(np.linspace(0, len(prompt_sizes) - 1, len(prompt_sizes)), labels=[str(x) for x in prompt_sizes], fontsize=12)
plt.savefig(average_fig_path)
plt.close()
