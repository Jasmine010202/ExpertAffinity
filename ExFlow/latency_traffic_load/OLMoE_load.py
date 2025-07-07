import numpy as np
import matplotlib.pyplot as plt
import os
import json

model_name = "OLMoE"#Switch_Transformer OLMoE
task_name = "generate"
input_name = "sonnet"
#phrase_mode = "decode" #decode
use_bipart = False  #True False
prompt_nums = [512] # 8, 16, 32, 64, 128, 256, 512, 1024
top_k = 8 # ST:1,OL:8

num_of_nodes = 2
num_of_gpus_pre_node = 2
#gpu_node_mapping = {0: 0, 1: 0, 2: 1, 3: 1}  # GPU 0和1映射到node 0；2和3映射到node 1
gpu_node_mapping = [0,0,1,1]

num_layers = 16 #decode only
batch_size=64
num_of_experts_per_layer = 64

Byte_per_token = 3072 #1个token占的字节

fig_dir = f"load/{model_name}/{task_name}_{input_name}/figs"
os.makedirs(fig_dir, exist_ok=True)

result_dir = f"load/{model_name}/{task_name}_{input_name}/data"
os.makedirs(result_dir, exist_ok=True)


# def experts_activations_count(routing_trace):
#     experts_activation_stats = np.zeros((num_layers, num_of_experts_per_layer)) 
#     for prompt in routing_trace:
#         token_traces = np.array(prompt["trace"])
#         num_tokens = token_traces.shape[0]

#         for token_id in range(num_tokens):
#             for layer_id in range(num_layers):
#                 expert_id = token_traces[token_id, layer_id]

#                 # fake
#                 # experts_activation_stats[layer_id,expert_id]  += 1

#                 if expert_id == -1:
#                     continue
#                 else:
#                     experts_activation_stats[layer_id,expert_id]  += 1

#     return experts_activation_stats

def experts_activations_count(routing_trace):
    num_tokens = routing_trace.shape[0]
    experts_activation_stats = np.zeros((num_layers, num_of_experts_per_layer)) 

    for token in range(num_tokens):
        for layer in range(num_layers):
            for expert in routing_trace[token,layer]:
                experts_activation_stats[layer,expert]  += 1
    return experts_activation_stats

def convert_placement_json_to_matrix(expert_placement_dict):
    expert_placement_matrix = np.full((num_layers, num_of_experts_per_layer), fill_value=-1, dtype=int)

    for layer_idx in range(num_layers):
        layer_key = f"layer_{layer_idx}"
        gpu_map = expert_placement_dict[layer_key]
        for gpu_id_str, expert_list in gpu_map.items():
            gpu_id = int(gpu_id_str)
            for expert_id in expert_list:
                expert_placement_matrix[layer_idx, expert_id] = gpu_id
    return expert_placement_matrix


def generate_vanilla_placement_matrix():
    num_gpus = num_of_nodes * num_of_gpus_pre_node
    vanilla_placement_matrix = np.zeros((num_layers, num_of_experts_per_layer),dtype=int)
    experts_per_gpu = num_of_experts_per_layer // num_gpus

    for layer_id in range(num_layers):
        for gpu_id in range(num_gpus):
            start = gpu_id * experts_per_gpu
            end = (gpu_id + 1) * experts_per_gpu
            vanilla_placement_matrix[layer_id, start:end] = gpu_id

    return vanilla_placement_matrix


def calculate_load(num_of_prompts, expert_placement, experts_activation_stats, type):
    # num_layers, num_experts = experts_activation_stats.shape
    num_gpus = num_of_nodes * num_of_gpus_pre_node
    
    gpus_load = {gpu_id:0 for gpu_id in range(num_gpus)}
    nodes_load = {node_id:0 for node_id in range(num_of_nodes)}

    gpus_token_num = {gpu_id:0 for gpu_id in range(num_gpus)}
    nodes_token_num = {node_id:0 for node_id in range(num_of_nodes)}
    
    # 每个 GPU 的总负载
    for layer_id in range(num_layers):
        for expert_id in range(num_of_experts_per_layer):
            expert_gpu_id = expert_placement[layer_id, expert_id]
            expert_activation_count = experts_activation_stats[layer_id, expert_id]
            expert_load = expert_activation_count * Byte_per_token # 字节数
            gpus_load[expert_gpu_id] += expert_load # 字节数
            gpus_token_num[expert_gpu_id] += expert_activation_count # token数

    # 每个节点的总负载
    for gpu_id, gpu_load in gpus_load.items():
        node_id = gpu_node_mapping[gpu_id]
        nodes_load[node_id] += gpu_load

    # 每个节点的总负载
    for gpu_id, gpu_num in gpus_token_num.items():
        node_id = gpu_node_mapping[gpu_id]
        nodes_token_num[node_id] += gpu_num

    load_per_gpu = {}
    for gpu_id, gpu_load in gpus_load.items():
        load_per_gpu[str(gpu_id)] = {
            "token_num": gpus_token_num[gpu_id],
            "bytes":gpu_load,
            "KB":gpu_load/1024,
            "MB":gpu_load/(1024*1024)}
    
    load_per_node = {}
    for node_id, node_load in nodes_load.items():
        load_per_node[str(node_id)] = {
            "token_num": nodes_token_num[node_id],
            "bytes":node_load,
            "KB":node_load/1024,
            "MB":node_load/(1024*1024)}


    result={"gpus":load_per_gpu, "nodes":load_per_node}

    filename = os.path.join(result_dir, f"load_{num_of_prompts}_{type}.json")
    with open (filename, "w") as f:
        json.dump(result,f,indent=2)
    
    return result

def plot_load_compare(vanilla_result, exflow_result, fig_path):
    gpu_ids = sorted(vanilla_result["gpus"].keys(), key=lambda x: int(x))
    node_ids = sorted(vanilla_result["nodes"].keys(), key=lambda x: int(x))


    x_gpu = np.arange(len(gpu_ids))
    x_node = np.arange(len(node_ids))

    vanilla_gpu_load = [vanilla_result["gpus"][gid]["token_num"] for gid in gpu_ids]
    exflow_gpu_load = [exflow_result["gpus"][gid]["token_num"] for gid in gpu_ids]

    vanilla_node_load = [vanilla_result["nodes"][nid]["token_num"] for nid in node_ids]
    exflow_node_load = [exflow_result["nodes"][nid]["token_num"] for nid in node_ids]

    width = 0.35
    color_vanilla = "#7fcdbb"
    color_exflow = "#1d91c0"

    # 创建画布
    title = "GPU & Node Load Comparison"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 2]})
    fig.suptitle(title, fontsize=14)
    fig.subplots_adjust(wspace=0.5)
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [2, 2]})
    # fig.suptitle(title, fontsize=14)
    # fig.subplots_adjust(wspace=0.4)


    # --- GPU 子图 ---
    ax1.ticklabel_format(style='plain', axis='y')
    ax1.bar(x_gpu - width / 2, vanilla_gpu_load, width=width, label="Vanilla", alpha=0.9, color=color_vanilla)
    ax1.bar(x_gpu + width / 2, exflow_gpu_load, width=width, label="Affinity", alpha=0.9, color=color_exflow)
    ax1.set_xlabel("GPU ID", fontsize=12)
    ax1.set_ylabel("Num of Tokens", fontsize=12)
    #ax1.set_title("Per-GPU Load", fontsize=12)
    ax1.set_xticks(x_gpu)
    ax1.set_xticklabels(gpu_ids)
    #ax1.grid(axis='y', linestyle='--', alpha=0.4)

    for i, val in enumerate(vanilla_gpu_load):
        ax1.text(x_gpu[i] - width / 2, val + 0.05, f"{val:.0f}", ha='center', va='bottom', fontsize=8, rotation=0)
    for i, val in enumerate(exflow_gpu_load):
        ax1.text(x_gpu[i] + width / 2, val + 0.05, f"{val:.0f}", ha='center', va='bottom', fontsize=8, rotation=0)

    # --- Node 子图 ---
    ax2.ticklabel_format(style='plain', axis='y') 
    ax2.bar(x_node - width / 2, vanilla_node_load, width=width, label="Vanilla", alpha=0.9, color=color_vanilla)
    ax2.bar(x_node + width / 2, exflow_node_load, width=width, label="Affinity", alpha=0.9, color=color_exflow)
    ax2.set_xlabel("Node ID", fontsize=12)
    #ax2.set_ylabel("Num of Tokens", fontsize=12)
    #ax2.set_title("Per-Node Load", fontsize=12)
    ax2.set_xticks(x_node)
    ax2.set_xticklabels(node_ids)
    #ax2.grid(axis='y', linestyle='--', alpha=0.4)

    for i, val in enumerate(vanilla_node_load):
        ax2.text(x_node[i] - width / 2, val + 0.05, f"{val:.0f}", ha='center', va='bottom', fontsize=8, rotation=0)
    for i, val in enumerate(exflow_node_load):
        ax2.text(x_node[i] + width / 2, val + 0.05, f"{val:.0f}", ha='center', va='bottom', fontsize=8, rotation=0)

    # --- 图例放在左侧外部 ---
    ax1.legend(loc="upper right", fontsize=10)

    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    plt.savefig(fig_path)
    plt.close()


if __name__ == "__main__":

    for num in prompt_nums:

        # routing_trace = []
        # with open(f"../different_tasks/trace_by_prompt/expert_trace/{model_name}/{task_name}/{input_name}/routing_trace_{num}_back.jsonl", 'r') as f:
        #     for line in f:
        #         entry = json.loads(line)
        #         prompt_id = entry["prompt_id"]
        #         trace = entry["trace"]  # shape: [num_tokens, num_layers]
        #         routing_trace.append({
        #             "prompt_id": prompt_id,
        #             "trace": trace
        #         })
        
        routing_trace = np.load(f'../expert_trace/{model_name}/{input_name}/top{top_k}/decode_routing_trace_{num}.npy')

        with open("../cluster_result/spectral/balance/clusters_result_all_layers.json", "r") as f:
            affinity_placement_dict = json.load(f)

        affinity_placement = convert_placement_json_to_matrix(affinity_placement_dict)
        # print(affinity_placement.shape)
        # print(affinity_placement)
        vanilla_placement = generate_vanilla_placement_matrix()
        # print(vanilla_placement.shape)
        # print(vanilla_placement)
        
        experts_activation_stats = experts_activations_count(routing_trace)
        #print(experts_activation_stats.shape)
        
        vanilla_result = calculate_load(num, vanilla_placement, experts_activation_stats, "vanilla")
        affinity_result = calculate_load(num, affinity_placement, experts_activation_stats, "affinity")

        fig_path = os.path.join(fig_dir,f"load_compare_{num}_token_name.png")
        plot_load_compare(vanilla_result, affinity_result, fig_path)

