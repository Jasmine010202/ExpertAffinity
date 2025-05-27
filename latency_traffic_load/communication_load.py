import numpy as np
import matplotlib.pyplot as plt
import os
import json
from matplotlib.patches import Patch

model_name = "Switch_Transformer"#Switch_Transformer OLMoE
task_name = "generate"
input_name = "sonnet"
#phrase_mode = "decode" #decode
use_bipart = False  #True False
prompt_nums = [512] # 8, 16, 32, 64, 128, 256, 512, 1024
# top_k = 1 # ST:1,OL:8

num_of_nodes = 2
num_of_gpus_pre_node = 2
#gpu_node_mapping = {0: 0, 1: 0, 2: 1, 3: 1}  # GPU 0和1映射到node 0；2和3映射到node 1
gpu_node_mapping = [0,0,1,1]

num_layers = 12 #6个encode+6个decode
batch_size=64

Byte_per_token = 3072 #1个token占的字节

fig_dir = f"communication_load/{model_name}/{task_name}_{input_name}/figs"
os.makedirs(fig_dir, exist_ok=True)

result_dir = f"communication_load/{model_name}/{task_name}_{input_name}/data"
os.makedirs(result_dir, exist_ok=True)

# Map prompt_id to GPU
def prompt_to_gpu(prompt_id):
    if prompt_id < 64 or (256 <= prompt_id < 320):
        return 0
    elif 64 <= prompt_id < 128 or (320 <= prompt_id < 384):
        return 1
    elif 128 <= prompt_id < 192 or (384 <= prompt_id < 448):
        return 2
    else:
        return 3

def calculate_communitcation_load(num_of_prompts,routing_trace, expert_placement,type):
    num_gpus = num_of_nodes * num_of_gpus_pre_node

    send_load_gpu = {gpu_id: 0 for gpu_id in range(num_gpus)}
    recv_load_gpu = {gpu_id: 0 for gpu_id in range(num_gpus)}

    send_load_node = {node_id: 0 for node_id in range(num_of_nodes)}
    recv_load_node = {node_id: 0 for node_id in range(num_of_nodes)}

    for prompt in routing_trace:
        # token 所在GPU和节点
        token_gpu_id = prompt_to_gpu(prompt["prompt_id"])
        token_node_id = gpu_node_mapping[token_gpu_id]
        
        token_traces = np.array(prompt["trace"])
        num_tokens = token_traces.shape[0]

        for token_id in range(num_tokens):
            
            if type=="exflow" or type == "token_context":
                prev_gpu_id = token_gpu_id  # 初始：token 所在 GPU
                prev_node_id = token_node_id

            for layer_id in range(num_layers):  
                expert_id = token_traces[token_id, layer_id]
                if expert_id == -1:
                    continue

                expert_gpu_id = expert_placement[layer_id, expert_id]
                expert_node_id = gpu_node_mapping[expert_gpu_id]

                if type=="vanilla" or type=="affinity_placement":
                    if expert_node_id == token_node_id and expert_gpu_id != token_gpu_id:
                        # 分发
                        send_load_gpu[token_gpu_id] += 1
                        recv_load_gpu[expert_gpu_id] += 1
                        # 聚合
                        send_load_gpu[expert_gpu_id] += 1
                        recv_load_gpu[token_gpu_id] += 1
                    elif expert_node_id != token_node_id:
                        send_load_node[token_node_id] += 1
                        recv_load_node[expert_node_id] += 1

                        send_load_node[expert_node_id] += 1
                        recv_load_node[token_node_id] += 1
                else:
                    if expert_node_id == prev_node_id and expert_gpu_id != prev_gpu_id:
                        send_load_gpu[prev_gpu_id] += 1
                        recv_load_gpu[expert_gpu_id] += 1
                    elif expert_node_id != prev_node_id:
                        send_load_node[prev_node_id] += 1
                        recv_load_node[expert_node_id] += 1
                    
                    prev_gpu_id = expert_gpu_id
                    prev_node_id = expert_node_id
    
    result = {
        "gpu_send_tokens": {str(k): v for k, v in send_load_gpu.items()},
        "gpu_recv_tokens": {str(k): v for k, v in recv_load_gpu.items()},
        "node_send_tokens": {str(k): v for k, v in send_load_node.items()},
        "node_recv_tokens": {str(k): v for k, v in recv_load_node.items()}
    }

    filename = os.path.join(result_dir, f"communication_load_{num_of_prompts}_{type}.json")
    with open (filename, "w") as f:
        json.dump(result,f,indent=2)

    return result

def calculate_communitcation_load_only_gpu(num_of_prompts,routing_trace, expert_placement,type):
    num_gpus = num_of_nodes * num_of_gpus_pre_node

    send_load_gpu = {gpu_id: 0 for gpu_id in range(num_gpus)}
    recv_load_gpu = {gpu_id: 0 for gpu_id in range(num_gpus)}

    for prompt in routing_trace:
        # token 所在GPU和节点
        token_gpu_id = prompt_to_gpu(prompt["prompt_id"])
        
        token_traces = np.array(prompt["trace"])
        num_tokens = token_traces.shape[0]

        for token_id in range(num_tokens):
            
            if type=="exflow" or type == "token_context":
                prev_gpu_id = token_gpu_id  # 初始：token 所在 GPU

            for layer_id in range(num_layers):  
                expert_id = token_traces[token_id, layer_id]
                if expert_id == -1:
                    continue

                expert_gpu_id = expert_placement[layer_id, expert_id]
            
                if type=="vanilla" or type=="affinity_placement":
                    if expert_gpu_id != token_gpu_id:
                        # 分发
                        send_load_gpu[token_gpu_id] += 1
                        recv_load_gpu[expert_gpu_id] += 1
                        # 聚合
                        send_load_gpu[expert_gpu_id] += 1
                        recv_load_gpu[token_gpu_id] += 1
                else:
                    if expert_gpu_id != prev_gpu_id:
                        send_load_gpu[prev_gpu_id] += 1
                        recv_load_gpu[expert_gpu_id] += 1
                    
                    prev_gpu_id = expert_gpu_id
    
    result = {
        "gpu_send_tokens": {str(k): v for k, v in send_load_gpu.items()},
        "gpu_recv_tokens": {str(k): v for k, v in recv_load_gpu.items()}
    }

    filename = os.path.join(result_dir, f"communication_load_{num_of_prompts}_{type}.json")
    with open (filename, "w") as f:
        json.dump(result,f,indent=2)

    return result

def plot_communication_compare(vanilla_result, context_result, exflow_result, affinity_result, fig_path, mode):
    assert mode in ["gpu", "node"], "mode must be 'gpu' or 'node'"

    methods = ['Vanilla', 'Token_Context', 'ExFlow', 'Affinity_Placement']
    results = [vanilla_result, context_result, exflow_result, affinity_result]
    colors = ['#a6d9c7', '#7fcdbb', '#41b6c4', '#1d91c0']

    send_key = f"{mode}_send_tokens"
    recv_key = f"{mode}_recv_tokens"

    ids = sorted(results[0][send_key].keys(), key=lambda x: int(x))
    x = np.arange(len(ids))  # 每个GPU或Node的x轴位置

    fig, ax = plt.subplots(figsize=(18, 6))

    bar_width = 0.08
    group_width = 8 * bar_width  # 每个 GPU/Node 一组 8 个柱子

    for i, (method, color, result) in enumerate(zip(methods, colors, results)):
        send_vals = [result[send_key][id_] for id_ in ids]
        recv_vals = [result[recv_key][id_] for id_ in ids]

        send_pos = x - group_width/2 + (2 * i) * bar_width
        recv_pos = x - group_width/2 + (2 * i + 1) * bar_width

        send_bars = ax.bar(send_pos, send_vals, width=bar_width, label=f"{method} Send", color=color)
        recv_bars = ax.bar(recv_pos, recv_vals, width=bar_width, label=f"{method} Recv", color=color, hatch='//')

        # 添加数值标签
        for bar in send_bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{int(height)}", 
                        ha='center', va='bottom', fontsize=7, rotation=0)
        for bar in recv_bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{int(height)}", 
                        ha='center', va='bottom', fontsize=7, rotation=0)


    ax.set_xlabel(f"{mode.upper()} ID")
    ax.set_ylabel("Num of Tokens")
    ax.set_title(f"{mode.upper()} Send/Recv Tokens Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=0)
    #ax.grid(axis="y", linestyle="--", alpha=0.4)

    # 自定义图例，只显示一次每种方法（用send表示），hatch代表recv
    
    legend_elements = [Patch(facecolor=color, label=method) for color, method in zip(colors, methods)]
    legend_elements.append(Patch(facecolor='white', edgecolor='black', label='Send'))
    legend_elements.append(Patch(facecolor='white', edgecolor='black', hatch='//', label='Receive'))
    
    fig.subplots_adjust(top=0.78)  # 默认大概是 0.9~0.95，调小一点可以腾出空间
    
    if mode == "gpu":
        loaction = "upper left"
    else:
        loaction = "upper center"
    ax.legend(loc=loaction, handles=legend_elements, ncol=1, fontsize=9)

    #plt.tight_layout()
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()




if __name__ == "__main__":

    for num in prompt_nums:

        routing_trace = []
        with open(f"../different_tasks/trace_by_prompt/expert_trace/{model_name}/{task_name}/{input_name}/routing_trace_{num}_back.jsonl", 'r') as f:
            for line in f:
                entry = json.loads(line)
                prompt_id = entry["prompt_id"]
                trace = entry["trace"]  # shape: [num_tokens, num_layers]
                routing_trace.append({
                    "prompt_id": prompt_id,
                    "trace": trace
                })
        #routing_trace = np.load(f"../different_tasks/trace_by_prompt/expert_trace/{model_name}/{task_name}/{input_name}/encode_routing_trace_{num}_front.npy")

        vanilla_encode_placement = np.load(f"../different_tasks/expert_placement/{model_name}/{input_name}/{'use_bipart' if use_bipart else 'not_use_bipart'}/encode/intra{num_of_gpus_pre_node}_inter{num_of_nodes}_vanilla.npy")
        vanilla_decode_placement = np.load(f"../different_tasks/expert_placement/{model_name}/{input_name}/{'use_bipart' if use_bipart else 'not_use_bipart'}/decode/intra{num_of_gpus_pre_node}_inter{num_of_nodes}_vanilla.npy")
        vanilla_placement = np.concatenate((vanilla_encode_placement, vanilla_decode_placement), axis=0)
        
        exflow_encode_placement = np.load(f"../different_tasks/expert_placement/{model_name}/{input_name}/{'use_bipart' if use_bipart else 'not_use_bipart'}/encode/{num}_front/intra{num_of_gpus_pre_node}_inter{num_of_nodes}.npy")
        exflow_decode_placement = np.load(f"../different_tasks/expert_placement/{model_name}/{input_name}/{'use_bipart' if use_bipart else 'not_use_bipart'}/decode/{num}_front/intra{num_of_gpus_pre_node}_inter{num_of_nodes}.npy")
        exflow_placement = np.concatenate((exflow_encode_placement, exflow_decode_placement), axis=0)

        vanilla_result=calculate_communitcation_load(num, routing_trace,vanilla_placement,"vanilla")
        exflow_result=calculate_communitcation_load(num, routing_trace, exflow_placement,"exflow")
        affinity_result=calculate_communitcation_load(num, routing_trace, exflow_placement,"affinity_placement")
        context_result = calculate_communitcation_load(num, routing_trace,vanilla_placement,"token_context")

        # vanilla_result=calculate_communitcation_load_only_gpu(num, routing_trace,vanilla_placement,"vanilla")
        # exflow_result=calculate_communitcation_load_only_gpu(num, routing_trace, exflow_placement,"exflow")
        # affinity_result=calculate_communitcation_load_only_gpu(num, routing_trace, exflow_placement,"affinity_placement")
        # context_result = calculate_communitcation_load_only_gpu(num, routing_trace,vanilla_placement,"token_context")

        fig_path = os.path.join(fig_dir,f"communication_load_{num}_4_token_gpu.png")
        # plot_traffic_compare(vanilla_result, exflow_result, fig_path)
        #plot_traffic_compare(vanilla_result, affinity_result, fig_path)
        #plot_traffic_compare_2(vanilla_result, exflow_result, affinity_result, fig_path)
        plot_communication_compare(vanilla_result, context_result, exflow_result, affinity_result, fig_path, mode="gpu")
        fig_path = os.path.join(fig_dir,f"communication_load_{num}_4_token_node.png")
        plot_communication_compare(vanilla_result, context_result, exflow_result, affinity_result, fig_path, mode="node")