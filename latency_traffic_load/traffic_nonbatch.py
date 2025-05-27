import numpy as np
import matplotlib.pyplot as plt
import os
import json

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

fig_dir = f"traffic/{model_name}/{task_name}_{input_name}/non_batch/4/figs"
os.makedirs(fig_dir, exist_ok=True)

result_dir = f"traffic/{model_name}/{task_name}_{input_name}/non_batch/4/data"
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

def calculate_traffic(num_of_prompts,routing_trace, expert_placement,type):
    
    num_intra_gpu = 0  # GPU 内通信
    num_inter_gpu = 0  # GPU 间通信(节点内)
    num_inter_node = 0  # 节点间通信

    num_layers = expert_placement.shape[0]
    #num_batches = num_of_prompts // batch_size # batch数

    result = {}

    for prompt in routing_trace:
        # token 所在GPU和节点
        token_gpu_id = prompt_to_gpu(prompt["prompt_id"])
        token_node_id = gpu_node_mapping[token_gpu_id]
        #print(f"token_gpu:{token_gpu_id};node:{token_node_id}")

        token_traces = np.array(prompt["trace"])
        num_tokens = token_traces.shape[0]
        #print(f"token_traces:{token_traces.shape}")

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
                    if expert_gpu_id == token_gpu_id:
                        num_intra_gpu +=2
                    elif expert_node_id == token_node_id and expert_gpu_id != token_gpu_id:
                        num_inter_gpu +=2
                    else:
                        num_inter_node +=2
                    
                else:
                    if expert_gpu_id == prev_gpu_id:
                        num_intra_gpu +=1
                    elif expert_node_id == prev_node_id and expert_gpu_id != prev_gpu_id:
                        num_inter_gpu +=1
                    else:
                        num_inter_node +=1
                
                    prev_gpu_id = expert_gpu_id
                    prev_node_id = expert_node_id
    
    traffic_intra_gpu = num_intra_gpu * Byte_per_token
    traffic_inter_gpu = num_inter_gpu * Byte_per_token
    traffic_inter_node = num_inter_node * Byte_per_token

    traffic_intra_gpu_KB = traffic_intra_gpu / 1024
    traffic_inter_gpu_KB = traffic_inter_gpu / 1024
    traffic_inter_node_KB = traffic_inter_node / 1024

    traffic_intra_gpu_MB = traffic_intra_gpu / (1024 * 1024)
    traffic_inter_gpu_MB = traffic_inter_gpu / (1024 * 1024)
    traffic_inter_node_MB = traffic_inter_node / (1024 * 1024)

    print(f"{type}")
    print(f"GPU内通信次数:\t{num_intra_gpu}")
    print(f"跨GPU通信次数:\t{num_inter_gpu}")
    print(f"跨节点通信次数:\t{num_inter_node}")

    print(f"GPU内通信字节数:\t{traffic_intra_gpu}")
    print(f"跨GPU通信字节数:\t{traffic_inter_gpu}")
    print(f"跨节点通信字节数:\t{traffic_inter_node}")

    print(f"GPU内通信量_KB:\t{traffic_intra_gpu_KB}")
    print(f"跨GPU通信量_KB:\t{traffic_inter_gpu_KB}")
    print(f"跨节点通信量_KB:\t{traffic_inter_node_KB}")

    print(f"GPU内通信量_MB:\t{traffic_intra_gpu_MB}")
    print(f"跨GPU通信量_MB:\t{traffic_inter_gpu_MB}")
    print(f"跨节点通信量_MB:\t{traffic_inter_node_MB}")
    
    result["num_intra_gpu"]=num_intra_gpu
    result["num_inter_gpu"]=num_inter_gpu
    result["num_inter_node"]=num_inter_node
    result["traffic_intra_gpu_Byte"]=traffic_intra_gpu
    result["traffic_inter_gpu_Byte"]=traffic_inter_gpu
    result["traffic_inter_node_Byte"]=traffic_inter_node
    result["traffic_intra_gpu_KB"]=traffic_intra_gpu_KB
    result["traffic_inter_gpu_KB"]=traffic_inter_gpu_KB
    result["traffic_inter_node_KB"]=traffic_inter_node_KB
    result["traffic_intra_gpu_MB"]=traffic_intra_gpu_MB
    result["traffic_inter_gpu_MB"]=traffic_inter_gpu_MB
    result["traffic_inter_node_MB"]=traffic_inter_node_MB

    filename = os.path.join(result_dir, f"traffic_{num_of_prompts}_{type}.json")
    with open (filename, "w") as f:
        json.dump(result,f,indent=2)

    return result

def plot_traffic_compare(vanilla_result, exflow_result, fig_path):
    labels = ["Inter-GPU", "Inter-Node"]
    x = np.arange(len(labels))
    width = 0.35

    # 提取 MB 通信量
    vanilla_values = [
        vanilla_result["traffic_inter_gpu_MB"],
        vanilla_result["traffic_inter_node_MB"]
    ]
    exflow_values = [
        exflow_result["traffic_inter_gpu_MB"],
        exflow_result["traffic_inter_node_MB"]
    ]

    # 颜色
    color_vanilla = "#7fcdbb"
    color_exflow = "#1d91c0"
    title="Communication Traffic Comparison"

    # 创建图形
    plt.figure(figsize=(8, 6))
    plt.bar(x - width / 2, vanilla_values, width=width, label="Vanilla", color=color_vanilla)
    plt.bar(x + width / 2, exflow_values, width=width, label="ExFlow", color=color_exflow)

    plt.xticks(x, labels)
    plt.ylabel("Traffic (MB)")
    plt.title(title)
    #plt.grid(axis="y", linestyle="--", alpha=0.4)

    # 柱顶加数值标签（横向）
    for i, val in enumerate(vanilla_values):
        plt.text(x[i] - width / 2, val + 0.5, f"{val:.2f}", ha='center', fontsize=9, rotation=0)

    for i, val in enumerate(exflow_values):
        plt.text(x[i] + width / 2, val + 0.5, f"{val:.2f}", ha='center', fontsize=9, rotation=0)

    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

def plot_traffic_compare_2(vanilla_result, exflow_result, affinity_result, fig_path):
    labels = ["Inter-GPU", "Inter-Node"]
    x = np.arange(len(labels))
    width = 0.2

    # 提取 MB 通信量
    vanilla_values = [
        vanilla_result["traffic_inter_gpu_MB"],
        vanilla_result["traffic_inter_node_MB"]
    ]
    exflow_values = [
        exflow_result["traffic_inter_gpu_MB"],
        exflow_result["traffic_inter_node_MB"]
    ]
    affinity_values = [
        affinity_result["traffic_inter_gpu_MB"],
        affinity_result["traffic_inter_node_MB"]
    ]

    # 调整柱子位置
    x_vanilla = x - width
    x_exflow = x
    x_affinity = x + width

    # 颜色
    color_vanilla = "#7fcdbb"
    color_exflow = "#1d91c0"
    color_affinity ="#41b6c4"
    title="Communication Traffic Comparison"

    # 创建图形
    plt.figure(figsize=(8, 6))
    plt.bar(x_vanilla, vanilla_values, width=width, label="Vanilla", color=color_vanilla)
    plt.bar(x_exflow, exflow_values, width=width, label="ExFlow", color=color_exflow)
    plt.bar(x_affinity, affinity_values, width=width, label="Affinity_Placement", color=color_affinity)

    plt.xticks(x, labels)
    plt.ylabel("Traffic (MB)")
    plt.title(title)
    #plt.grid(axis="y", linestyle="--", alpha=0.4)


    # 柱顶加数值标签（横向）
    for i, val in enumerate(vanilla_values):
        plt.text(x_vanilla[i], val + 0.5, f"{val:.2f}", ha='center', fontsize=9, rotation=0)

    for i, val in enumerate(exflow_values):
        plt.text(x_exflow[i], val + 0.5, f"{val:.2f}", ha='center', fontsize=9, rotation=0)

    for i, val in enumerate(affinity_values):
        plt.text(x_affinity[i], val + 0.5, f"{val:.2f}", ha='center', fontsize=9, rotation=0)

    #plt.legend(loc="upper right", fontsize=10)
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()


def plot_traffic_compare_3(vanilla_result, context_result, exflow_result, affinity_result, fig_path):
    labels = ["Inter-GPU", "Inter-Node"]
    x = np.arange(len(labels))
    width = 0.2

    # 提取 MB 通信量
    vanilla_values = [
        vanilla_result["traffic_inter_gpu_MB"],
        vanilla_result["traffic_inter_node_MB"]
    ]
    context_values = [
        context_result["traffic_inter_gpu_MB"],
        context_result["traffic_inter_node_MB"]
    ]
    exflow_values = [
        exflow_result["traffic_inter_gpu_MB"],
        exflow_result["traffic_inter_node_MB"]
    ]
    affinity_values = [
        affinity_result["traffic_inter_gpu_MB"],
        affinity_result["traffic_inter_node_MB"]
    ]

    # 调整柱子位置
    x_vanilla = x - 1.5 * width
    x_context = x - 0.5 * width
    x_exflow = x + 0.5 * width
    x_affinity = x + 1.5 * width

    # 颜色
    color_vanilla = "#a6d9c7"
    color_context = "#7fcdbb"
    color_exflow = "#41b6c4"
    color_affinity = "#1d91c0"
    title="Communication Traffic Comparison"

    # 创建图形
    plt.figure(figsize=(9, 6))
    plt.bar(x_vanilla, vanilla_values, width=width, label="Vanilla", color=color_vanilla)
    plt.bar(x_context, context_values, width=width, label="Token_Context", color=color_context)
    plt.bar(x_exflow, exflow_values, width=width, label="ExFlow", color=color_exflow)
    plt.bar(x_affinity, affinity_values, width=width, label="Affinity_Placement", color=color_affinity)

    plt.xticks(x, labels)
    plt.ylabel("Traffic (MB)")
    plt.title(title)
    #plt.grid(axis="y", linestyle="--", alpha=0.4)


    # 柱顶加数值标签（横向）
    for i in range(len(labels)):
        plt.text(x_vanilla[i], vanilla_values[i] + 0.5, f"{vanilla_values[i]:.2f}", ha='center', fontsize=9)
        plt.text(x_context[i], context_values[i] + 0.5, f"{context_values[i]:.2f}", ha='center', fontsize=9)
        plt.text(x_exflow[i], exflow_values[i] + 0.5, f"{exflow_values[i]:.2f}", ha='center', fontsize=9)
        plt.text(x_affinity[i], affinity_values[i] + 0.5, f"{affinity_values[i]:.2f}", ha='center', fontsize=9)


    #plt.legend(loc="upper right", fontsize=10)
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.savefig(fig_path)
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

        vanilla_result=calculate_traffic(num, routing_trace,vanilla_placement,"vanilla")
        exflow_result=calculate_traffic(num, routing_trace, exflow_placement,"exflow")
        affinity_result=calculate_traffic(num, routing_trace, exflow_placement,"affinity_placement")
        context_result = calculate_traffic(num, routing_trace,vanilla_placement,"token_context")

        fig_path = os.path.join(fig_dir,f"traffic_compare_{num}_4_token.png")
        plot_traffic_compare(vanilla_result, exflow_result, fig_path)
        #plot_traffic_compare(vanilla_result, affinity_result, fig_path)
        #plot_traffic_compare_2(vanilla_result, exflow_result, affinity_result, fig_path)
        plot_traffic_compare_3(vanilla_result, context_result, exflow_result, affinity_result, fig_path)