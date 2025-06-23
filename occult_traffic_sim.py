import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import json


model_name = "OLMoE"
# input_name = "sonnet"   
prompt_nums = [512] # 8, 16, 32, 64, 128, 256, 512, 1024
top_k = 8 

# num_of_nodes = 2
# num_of_gpus_pre_node = 2

num_gpus =4

gpu_node_mapping = np.array([0, 0, 1, 1]) # GPU 0和1映射到node 0；2和3映射到node 1

num_layers = 16
num_experts_per_layer = 64

enable_occult = True
suffix = 'occult' if enable_occult else 'original'

# Byte_per_token = 4096 #1个token占的字节 

fig_dir = f"Occult_test/traffic_sim/traffic_test/{model_name}_top{top_k}/figs"
os.makedirs(fig_dir, exist_ok=True)

result_dir = f"Occult_test/traffic_sim/traffic_test/{model_name}_top{top_k}/data"
os.makedirs(result_dir, exist_ok=True)


def extract_routing_trace(file_path):
    routing_trace = []
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            prompt_id = entry["prompt_id"]
            trace = entry["trace"]  # shape: [num_tokens, num_layers]
            routing_trace.append({
                "prompt_id": prompt_id,
                "trace": trace
            })
    return routing_trace


def extract_expert_placement(file_path):
    with open(file_path,'r') as f:
        placement_dict = json.load(f)
    
    # 转成np.array  [layers,experts]
    expert_placement = np.full((num_layers, num_experts_per_layer), -1, dtype=int)
    for layer_str, gpu_experts_lists in placement_dict.items():
        layer_id = int(layer_str)
        for gpu_id, expert_list in enumerate(gpu_experts_lists):
            for expert_id in expert_list:
                expert_placement[layer_id, expert_id] = gpu_id

    return expert_placement


# Map prompt_id to GPU
# 模拟DP+EP batch_size=64 prompt_num = 512 单节点4张卡
def prompt_to_gpu(prompt_id):
    if prompt_id < 64 or (256 <= prompt_id < 320):
        return 0
    elif 64 <= prompt_id < 128 or (320 <= prompt_id < 384):
        return 1
    elif 128 <= prompt_id < 192 or (384 <= prompt_id < 448):
        return 2
    else:
        return 3


def calculate_num_of_token_copies(routing_trace, expert_placement):
    # 转发的token副本数
    num_intra_gpu = 0  # GPU 内
    num_inter_gpu = 0  # GPU 间
    num_inter_node = 0  # 节点间

    #num_layers = expert_placement.shape[0]
    result = {}

    for prompt in routing_trace:
        # token 所在GPU
        token_gpu_id = prompt_to_gpu(prompt["prompt_id"])
        #token_node_id = gpu_node_mapping[token_gpu_id]

        token_routing_traces = np.array(prompt["trace"])
        num_tokens = token_routing_traces.shape[0]

        for token_id in range(num_tokens):
            for layer_id in range(num_layers):  
                expert_ids = token_routing_traces[token_id, layer_id]
                expert_gpu_ids = expert_placement[layer_id, expert_ids]
                #expert_node_ids = gpu_node_mapping[expert_gpu_ids]
                # print(f"expert_gpu_ids:{expert_gpu_ids}")
                
                if enable_occult:
                    # 每个 GPU 接收一份副本
                    unique_gpu_ids = np.unique(expert_gpu_ids)  
                    # print(f"unique_gpu_ids:{unique_gpu_ids}")

                    if token_gpu_id in unique_gpu_ids:
                        intra_num = 1
                    else:
                        intra_num = 0
                    
                    inter_num = len(unique_gpu_ids) - intra_num
                else:
                    # 每个 GPU 上的每个专家都接收一个副本
                    gpu_ids = expert_gpu_ids
                    # print(f"gpu_ids:{gpu_ids}")
                    intra_num = int(np.sum(gpu_ids == token_gpu_id))
                    inter_num = int(np.sum(gpu_ids != token_gpu_id))


                # print(f"intra_num:{intra_num}")
                # print(f"inter_num:{inter_num}")

                num_intra_gpu += intra_num
                num_inter_gpu += inter_num

        #     break
        # break

    print(f"GPU内token副本数:\t{num_intra_gpu}")
    print(f"跨GPUtoken副本数:\t{num_inter_gpu}")
 
    result["num_intra_gpu"]=num_intra_gpu
    result["num_inter_gpu"]=num_inter_gpu
    return result


def plot_num_of_copies_compare(num_of_token_copies, fig_path):
    dataset_labels = ["sonnet", "GSM8K", "conala"]
    x = np.arange(len(dataset_labels))
    width = 0.2

    # 取3个数据集在4种放置下的跨GPU通信副本数
    vanilla = [num_of_token_copies[ds]["vanilla_placement"]["num_inter_gpu"] for ds in dataset_labels]
    sonnet_occult = [num_of_token_copies[ds]["sonnet_occult_placement"]["num_inter_gpu"] for ds in dataset_labels]
    GSM8K_occult = [num_of_token_copies[ds]["GSM8K_occult_placement"]["num_inter_gpu"] for ds in dataset_labels]
    conala_occult = [num_of_token_copies[ds]["conala_occult_placement"]["num_inter_gpu"] for ds in dataset_labels]

    # 调整柱子位置
    x_vanilla = x - 1.5 * width
    x_sonnet = x - 0.5 * width
    x_GSM8K = x + 0.5 * width
    x_conala = x + 1.5 * width

    # 颜色
    colors = ["#a6d9c7", "#7fcdbb", "#41b6c4", "#1d91c0"]

    # 创建图形
    plt.figure(figsize=(10, 6))
    plt.bar(x_vanilla, vanilla, width=width, label="vanilla", color=colors[0])
    plt.bar(x_sonnet, sonnet_occult, width=width, label="sonnet_occult", color=colors[1])
    plt.bar(x_GSM8K, GSM8K_occult, width=width, label="GSM8K_occult", color=colors[2])
    plt.bar(x_conala, conala_occult, width=width, label="conala_occult", color=colors[3])

    plt.xticks(x, dataset_labels)
    plt.ylabel("Num of Token Copies")
    plt.xlabel("Datasets (512 prompts)")
    plt.title("Comparison of Token Copies Transferred Across GPUs")
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    
    max_value = max(vanilla + sonnet_occult + GSM8K_occult + conala_occult)
    plt.ylim(0, max_value * 1.1)  

    offset = 10000
    # 柱顶加数值标签（横向）
    for i in range(len(dataset_labels)):
        plt.text(x_vanilla[i], vanilla[i] + offset, f"{vanilla[i]:.0f}", ha='center', fontsize=9)
        plt.text(x_sonnet[i], sonnet_occult[i] + offset, f"{sonnet_occult[i]:.0f}", ha='center', fontsize=9)
        plt.text(x_GSM8K[i], GSM8K_occult[i] + offset, f"{GSM8K_occult[i]:.0f}", ha='center', fontsize=9)
        plt.text(x_conala[i], conala_occult[i] + offset, f"{conala_occult[i]:.0f}", ha='center', fontsize=9)

    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()



if __name__ == "__main__":

    for num in prompt_nums:

        sonnet_routing_trace = extract_routing_trace(f"Occult_test/expert_trace/traffic_test/by_prompt/{model_name}_sonnet_top{top_k}/routing_trace_{num}.jsonl")
        GSM8K_routing_trace = extract_routing_trace(f"Occult_test/expert_trace/traffic_test/by_prompt/{model_name}_GSM8K_top{top_k}/routing_trace_{num}.jsonl")
        conala_routing_trace = extract_routing_trace(f"Occult_test/expert_trace/traffic_test/by_prompt/{model_name}_conala_top{top_k}/routing_trace_{num}.jsonl")

        vanilla_placement = extract_expert_placement("Occult_test/expert_placement/OLMoE_vanilla_placement.json")
        sonnet_placement = extract_expert_placement("Occult_test/expert_placement/OLMoE_sonnet_placement_512.json")
        GSM8K_placement = extract_expert_placement("Occult_test/expert_placement/OLMoE_GSM8K_placement_512.json")
        conala_placement = extract_expert_placement("Occult_test/expert_placement/OLMoE_conala_placement_512.json")
        
        sonnet_vanilla_copies = calculate_num_of_token_copies(sonnet_routing_trace, vanilla_placement)
        sonnet_s_occult_copies = calculate_num_of_token_copies(sonnet_routing_trace, sonnet_placement)
        sonnet_G_occult_copies = calculate_num_of_token_copies(sonnet_routing_trace, GSM8K_placement)
        sonnet_c_occult_copies = calculate_num_of_token_copies(sonnet_routing_trace, conala_placement)

        GSM8K_vanilla_copies = calculate_num_of_token_copies(GSM8K_routing_trace, vanilla_placement)
        GSM8K_s_occult_copies = calculate_num_of_token_copies(GSM8K_routing_trace, sonnet_placement)
        GSM8K_G_occult_copies = calculate_num_of_token_copies(GSM8K_routing_trace, GSM8K_placement)
        GSM8K_c_occult_copies = calculate_num_of_token_copies(GSM8K_routing_trace, conala_placement)

        conala_vanilla_copies = calculate_num_of_token_copies(conala_routing_trace, vanilla_placement)
        conala_s_occult_copies = calculate_num_of_token_copies(conala_routing_trace, sonnet_placement)
        conala_G_occult_copies = calculate_num_of_token_copies(conala_routing_trace, GSM8K_placement)
        conala_c_occult_copies = calculate_num_of_token_copies(conala_routing_trace, conala_placement)

        num_of_token_copies = {}
        num_of_token_copies["sonnet"] = {}
        num_of_token_copies["sonnet"]["vanilla_placement"] = sonnet_vanilla_copies
        num_of_token_copies["sonnet"]["sonnet_occult_placement"] = sonnet_s_occult_copies
        num_of_token_copies["sonnet"]["GSM8K_occult_placement"] = sonnet_G_occult_copies
        num_of_token_copies["sonnet"]["conala_occult_placement"] = sonnet_c_occult_copies

        num_of_token_copies["GSM8K"] = {}
        num_of_token_copies["GSM8K"]["vanilla_placement"] = GSM8K_vanilla_copies
        num_of_token_copies["GSM8K"]["sonnet_occult_placement"] = GSM8K_s_occult_copies
        num_of_token_copies["GSM8K"]["GSM8K_occult_placement"] = GSM8K_G_occult_copies
        num_of_token_copies["GSM8K"]["conala_occult_placement"] = GSM8K_c_occult_copies

        num_of_token_copies["conala"] = {}
        num_of_token_copies["conala"]["vanilla_placement"] = conala_vanilla_copies
        num_of_token_copies["conala"]["sonnet_occult_placement"] = conala_s_occult_copies
        num_of_token_copies["conala"]["GSM8K_occult_placement"] = conala_G_occult_copies
        num_of_token_copies["conala"]["conala_occult_placement"] = conala_c_occult_copies

        filename = os.path.join(result_dir, f"num_of_token_copies_{suffix}.json")
        with open (filename, "w") as f:
            json.dump(num_of_token_copies ,f,indent=2)

        # with open(f"Occult_test/traffic_sim/traffic_test/OLMoE_top8/data/num_of_token_copies_{suffix}.json",'r') as f:
        #     num_of_token_copies = json.load(f)

        fig_path = os.path.join(fig_dir,f"num_of_token_copies_compare_{suffix}.svg")
        plot_num_of_copies_compare(num_of_token_copies, fig_path)
       