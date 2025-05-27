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

#input_ids= torch.Size([1, 17]),dtype=torch.int64
#hidden_states= torch.Size([1, 17, 768]),dtype=torch.float32
Byte_per_token = 3072 #1个token占的字节

# fig_dir = f"traffic/{model_name}/{task_name}_{input_name}/batch/exflow/figs"
# os.makedirs(fig_dir, exist_ok=True)

result_dir = f"traffic/{model_name}/{task_name}_{input_name}/batch/data"
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

def calculate_num_of_communication(num_of_prompts,routing_trace, expert_placement,type):
    
    num_layers = expert_placement.shape[0]
    num_batches = num_of_prompts // batch_size # batch数

    result = {}
    
    for batch_id in range(num_batches):
        batch_start = batch_id * batch_size
        batch_end = batch_start + batch_size

        # 当前这一个batch的prompts
        batch_prompts = [p for p in routing_trace if batch_start <= p["prompt_id"] < batch_end]
        #print(f"batch_id{batch_id};batch_start{batch_start};batch_end{batch_end};prompts{len(batch_prompts)}")

        # 当前 batch 被放在哪个 GPU 上(DP)->token所属的GPU是哪个
        token_gpu_id = prompt_to_gpu(batch_start)
        token_node_id = gpu_node_mapping[token_gpu_id]
        # print(f"token_gpu:{token_gpu_id};node:{token_node_id}")

        # 记每一层发生的同节点跨GPU、跨节点通信次数
        layer_stats = [{"intra_gpu": 0, "inter_gpu": 0, "inter_node": 0} for _ in range(num_layers)]

        for prompt in batch_prompts:
            token_traces = np.array(prompt["trace"])
            #print(f"token_traces:{token_traces.shape}")
            num_tokens = token_traces.shape[0]

            for token_id in range(num_tokens):
                if type=="exflow":
                    prev_gpu_id = token_gpu_id  # 初始：token 所在 GPU
                    prev_node_id = token_node_id

                for layer_id in range(num_layers):  
                    expert_id = token_traces[token_id, layer_id]
                    if expert_id == -1:
                        continue

                    expert_gpu_id = expert_placement[layer_id, expert_id]
                    expert_node_id = gpu_node_mapping[expert_gpu_id]

                    if type=="vanilla":
                        if expert_gpu_id == token_gpu_id:
                            layer_stats[layer_id]["intra_gpu"]+=2
                        elif expert_node_id == token_node_id and expert_gpu_id != token_gpu_id:
                            layer_stats[layer_id]["inter_gpu"]+=2
                        else:
                            layer_stats[layer_id]["inter_node"]+=2
                    else:
                        if expert_gpu_id == prev_gpu_id:
                            layer_stats[layer_id]["intra_gpu"]+=1
                        elif expert_node_id == prev_node_id and expert_gpu_id != prev_gpu_id:
                            layer_stats[layer_id]["inter_gpu"]+=1
                        else:
                            layer_stats[layer_id]["inter_node"]+=1

                        prev_gpu_id = expert_gpu_id
                        prev_node_id = expert_node_id

        # 汇总这一个batch的同节点跨GPU、跨节点通信次数
        batch_summary = {
            "intra_gpu": sum(l["intra_gpu"] for l in layer_stats),
            "inter_gpu": sum(l["inter_gpu"] for l in layer_stats),
            "inter_node": sum(l["inter_node"] for l in layer_stats)
        }

        result[f"batch_{batch_id}"] = {
            "per_layer": layer_stats,
            "batch_summary": batch_summary
        }
    
    # 每个batch内的通信次数相加，汇总所有prompt推理时 同节点跨GPU、跨节点通信次数总和
    total_num = {
        "intra_gpu": sum(result[b]["batch_summary"]["intra_gpu"] for b in result),
        "inter_gpu": sum(result[b]["batch_summary"]["inter_gpu"] for b in result),
        "inter_node": sum(result[b]["batch_summary"]["inter_node"] for b in result)
    }

    result["total_num"] = total_num

    # 每类次数 * hidden_states大小 = 每一类的总通信量 字节数
    total_traffic_Byte = {
        "intra_gpu": total_num["intra_gpu"]*Byte_per_token,
        "inter_gpu": total_num["inter_gpu"]*Byte_per_token,
        "inter_node": total_num["inter_node"]*Byte_per_token
    }

    result["total_traffic_Byte"] = total_traffic_Byte

    total_traffic_KB = {
        "intra_gpu": total_traffic_Byte["intra_gpu"]/1024,
        "inter_gpu": total_traffic_Byte["inter_gpu"]/1024,
        "inter_node": total_traffic_Byte["inter_node"]/1024,
    }

    result["total_traffic_KB"] = total_traffic_KB

    total_traffic_MB = {
        "intra_gpu": total_traffic_Byte["intra_gpu"]/(1024*1024),
        "inter_gpu": total_traffic_Byte["inter_gpu"]/(1024*1024),
        "inter_node": total_traffic_Byte["inter_node"]/(1024*1024),
    }

    result["total_traffic_MB"] = total_traffic_MB


    print(total_num)
    print(total_traffic_Byte)
    print(total_traffic_KB)
    print(total_traffic_MB)
    
    filename = os.path.join(result_dir, f"traffic_{num_of_prompts}_{type}.json")
    with open (filename, "w") as f:
        json.dump(result,f,indent=2)

    return result


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

        vanilla_encode_placement = np.load(f"../different_tasks/expert_placement/{model_name}/{input_name}/{'use_bipart' if use_bipart else 'not_use_bipart'}/encode/intra{num_of_gpus_pre_node}_inter{num_of_nodes}_vanilla.npy")
        vanilla_decode_placement = np.load(f"../different_tasks/expert_placement/{model_name}/{input_name}/{'use_bipart' if use_bipart else 'not_use_bipart'}/decode/intra{num_of_gpus_pre_node}_inter{num_of_nodes}_vanilla.npy")
        
        vanilla_placement = np.concatenate((vanilla_encode_placement, vanilla_decode_placement), axis=0)
        calculate_num_of_communication(num, routing_trace,vanilla_placement,"vanilla")

        exflow_encode_placement = np.load(f"../different_tasks/expert_placement/{model_name}/{input_name}/{'use_bipart' if use_bipart else 'not_use_bipart'}/encode/{num}_front/intra{num_of_gpus_pre_node}_inter{num_of_nodes}.npy")
        exflow_decode_placement = np.load(f"../different_tasks/expert_placement/{model_name}/{input_name}/{'use_bipart' if use_bipart else 'not_use_bipart'}/decode/{num}_front/intra{num_of_gpus_pre_node}_inter{num_of_nodes}.npy")
        
        exflow_placement = np.concatenate((exflow_encode_placement, exflow_decode_placement), axis=0)
        calculate_num_of_communication(num, routing_trace, exflow_placement,"exflow")