'''
    现存问题：
    1. 带宽应该是随着发送token的数量变化的,现在是固定的
    2. 现在的代码对于2个节点,每个节点两张卡的情况是没问题的,但是当每个节点的GPU数量超过2就不太对劲了。
    现在的设定token只有3个选择,要么不转发本地GPU计算,要么发给同节点不同GPU[只有一个选择],要么发给另一个节点上的两个GPU[跨节点通信,也只有一个选择]
    如果单节点的GPU数量>2,或者节点数>2,那么发给同节点不同GPU,或者发给不同节点,就不止有一个选择,需要加以区分,因为通信是并发的
    然后看发给谁的延迟最大,分别作为跨GPU通信延迟或者跨节点通信延迟,再对两者比较得出这一层的延迟
    现在因为两种情况都只有一个选择,所以分batch算延迟的时候,一个batch的token都在一个GPU上,那么跨GPU就是往一个方向发,跨节点也是,并发的时候就是两个方向同时,谁的时延大就是总通信量
'''

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
gpu_node_mapping = [0, 0, 1, 1]

num_layers = 12 #6个encode+6个decode
batch_size=64

#input_ids= torch.Size([1, 17]),dtype=torch.int64
#hidden_states= torch.Size([1, 17, 768]),dtype=torch.float32
Byte_per_token = 3072 #1个token占的字节

# 实测带宽，需要确定
bw_inter_gpu = 1.73  # GB/s
bw_inter_node = 0.12  # GB/s

# fig_dir = f"latency/{model_name}/{task_name}_{input_name}/figs"
# os.makedirs(fig_dir, exist_ok=True)

result_dir = f"latency/{model_name}/{task_name}_{input_name}/data"
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


def estimate_latency(bytes, bandwidth_gbps):
    return bytes / (bandwidth_gbps * 1e9 / 8) * 1000  # ms

def calculate_latency(num_of_prompts,routing_trace, expert_placement,type):
    
    # num_layers = expert_placement.shape[0]
    num_batches = num_of_prompts // batch_size # batch数

    encode_layers = list(range(6))
    decode_layers = list(range(6, 12))

    # 仅exflow用
    prev_gpu_id = -1  # 初始：token 所在 GPU
    prev_node_id = -1

    result = {}
    
    # 模拟DP+EP，分batch计算
    for batch_id in range(num_batches):
        batch_start = batch_id * batch_size
        batch_end = batch_start + batch_size
        # 当前这一个batch的prompts 
        batch_prompts = [p for p in routing_trace if batch_start <= p["prompt_id"] < batch_end]

        # Switch_Transformer是encode+decode 时延得分开看
        ###############################################################################################
        # Encode 阶段（各层同步） 记每一层的跨节点和节点内GPU间通信量,最后叠加得到总的encode阶段时延
        encoder_layer_stats = []

        # 按层算延迟，只把前6层的取出来
        # 算出来encode阶段每一层的时延
        for layer_id in encode_layers:
            inter_gpu_bytes, inter_node_bytes = 0, 0
            intra_gpu_bytes = 0

            for prompt in batch_prompts:
                token_traces = np.array(prompt["trace"])
                num_tokens = token_traces.shape[0]

                token_gpu_id = prompt_to_gpu(prompt["prompt_id"])
                token_node_id = gpu_node_mapping[token_gpu_id]

                for token_id in range(num_tokens):
                    expert_id = token_traces[token_id, layer_id]
                    # 属于decode阶段的先不管
                    if expert_id == -1:
                        continue

                    expert_gpu_id = expert_placement[layer_id, expert_id]
                    expert_node_id = gpu_node_mapping[expert_gpu_id]

                    if type == "vanilla":
                        traffic = Byte_per_token * 2    # 发出去再聚合
                    elif type == "exflow":
                        traffic = Byte_per_token    # 发出去就地计算，不回发
                        prev_gpu_id = expert_gpu_id  # 记发到了哪个节点，decode用
                        prev_node_id = expert_node_id

                    # if expert_gpu_id != token_gpu_id:
                    #     if expert_node_id == token_node_id:
                    #         inter_gpu_bytes += traffic
                    #     else:
                    #         inter_node_bytes += traffic
                    
                    if expert_gpu_id == token_gpu_id:
                        intra_gpu_bytes += traffic
                    elif expert_node_id == token_node_id and expert_gpu_id != token_gpu_id:
                        inter_gpu_bytes += traffic
                    else:
                        inter_node_bytes += traffic

            latency_inter_gpu = estimate_latency(inter_gpu_bytes,bw_inter_gpu)
            latency_inter_node = estimate_latency(inter_node_bytes,bw_inter_node)
            
            layer_stats = {
                "layer":layer_id,
                "intra_gpu_bytes":intra_gpu_bytes,
                "inter_gpu_bytes":inter_gpu_bytes,
                "inter_node_bytes":inter_node_bytes,
                "latency_inter_gpu_ms":latency_inter_gpu,
                "latency_inter_node_ms":latency_inter_node,
                "latency_ms":max(latency_inter_gpu,latency_inter_node) #并行发送，取最大的作为这一层的时延
            }

            encoder_layer_stats.append(layer_stats)

        ###############################################################################################
        # Decode阶段： 逐个token生成，有的会提前计算
        # 按时间步计算，每个时间步，一个batch内的prompt同时生成一个token（还没有结束的）
        decoder_layer_stats_per_step = []
        decode_tokens_per_prompt = [[] for _ in batch_prompts] # 每个prompt开一个空列表，存放decode阶段的token的id（长短不一，可能提前结束）
        step = 0

        # 取出每个prompt的decode token id
        for i, prompt in enumerate(batch_prompts):
            token_traces = np.array(prompt["trace"])
            num_tokens = token_traces.shape[0]
            for token_id in range(num_tokens):
                if np.any(token_traces[token_id][6:] != -1):    # 后六层不是-1，代表是decode阶段的token，记录tokenid
                    decode_tokens_per_prompt[i].append(token_id)

        #print(decode_tokens_per_prompt)

        while any(decode_tokens_per_prompt):    # 只要有任何一个prompt的decode阶段token还有剩余
            inter_gpu_bytes_layers = [0] * 6
            inter_node_bytes_layers = [0] * 6
            intra_gpu_bytes_layers = [0] * 6

            for i, prompt in enumerate(batch_prompts):
                if not decode_tokens_per_prompt[i]: # 如果这个prompt推理完了（没有decode token了），跳过
                    continue

                # 还有decode阶段的token
                # 取最前面的一个token-pop
                token_id = decode_tokens_per_prompt[i].pop(0)
                token_traces = np.array(prompt["trace"])

                if type=="vanilla":
                    token_gpu_id = prompt_to_gpu(prompt["prompt_id"])   #对应的prompt在哪里，token就属于哪个GPU
                    token_node_id = gpu_node_mapping[token_gpu_id]

                # 逐层看这个token选择了哪个专家
                for j, layer_id in enumerate(decode_layers):
                    expert_id = token_traces[token_id, layer_id]
                    if expert_id == -1:
                        continue

                    expert_gpu_id = expert_placement[layer_id, expert_id]
                    expert_node_id = gpu_node_mapping[expert_gpu_id]

                    if type == "vanilla":
                        traffic = Byte_per_token * 2    # 发出去再聚合
                        # if expert_gpu_id != token_gpu_id:   # 和token所在gpu比较
                        #     if expert_node_id == token_node_id:
                        #         inter_gpu_bytes_layers[j] += traffic
                        #     else:
                        #         inter_node_bytes_layers[j] += traffic
                        
                        if expert_gpu_id == token_gpu_id:
                            intra_gpu_bytes_layers[j] += traffic
                        elif expert_node_id == token_node_id and expert_gpu_id != token_gpu_id:
                            inter_gpu_bytes_layers[j] += traffic
                        else:
                            inter_node_bytes_layers[j] += traffic

                    elif type == "exflow":
                        traffic = Byte_per_token    # 发出去就地计算，不回发
                        # if expert_gpu_id != prev_gpu_id:
                        #     if expert_node_id == prev_node_id:
                        #         inter_gpu_bytes_layers[j] += traffic
                        #     else:
                        #         inter_node_bytes_layers[j] += traffic
                        
                        if expert_gpu_id == prev_gpu_id:
                            intra_gpu_bytes_layers[j] += traffic
                        elif expert_node_id == prev_node_id and expert_gpu_id != prev_gpu_id:
                            inter_gpu_bytes_layers[j] += traffic
                        else:
                            inter_node_bytes_layers[j] += traffic

                        prev_gpu_id = expert_gpu_id  # 更新所在位置发到了哪个节点
                        prev_node_id = expert_node_id

            # 所有prompt 的token分别 逐层看完，统计生成这一批token，各层的时延
            step_stats = []
            for j, layer_id in enumerate(decode_layers):
                latency_inter_gpu = estimate_latency(inter_gpu_bytes_layers[j],bw_inter_gpu)
                latency_inter_node = estimate_latency(inter_node_bytes_layers[j],bw_inter_node)
            
                layer_stats = {
                    "layer":layer_id,
                    "intra_gpu_bytes":intra_gpu_bytes_layers[j],
                    "inter_gpu_bytes":inter_gpu_bytes_layers[j],
                    "inter_node_bytes":inter_node_bytes_layers[j],
                    "latency_inter_gpu_ms":latency_inter_gpu,
                    "latency_inter_node_ms":latency_inter_node,
                    "latency_ms":max(latency_inter_gpu,latency_inter_node) #并行发送，取最大的作为这一层的时延
                }
                step_stats.append(layer_stats) # 存这一步所有 decode层

            decoder_layer_stats_per_step.append(step_stats)
            step += 1

        result[f"batch_{batch_id}"]=  {
            "encode": encoder_layer_stats,
            "decode": decoder_layer_stats_per_step
        }
        #print(f"decode_step:{step}")
    
    filename = os.path.join(result_dir, f"latency_statistics_per_batch_{num_of_prompts}_{type}.json")
    with open (filename, "w") as f:
        json.dump(result,f,indent=2)

    return result


def summarize_latency(num_of_prompts,latency_result,type):
    summary = {}
    # total_inter_gpu_latency = 0
    # total_inter_node_latency = 0

    for batch_key, batch_data in latency_result.items():

        total_inter_gpu_latency = 0
        total_inter_node_latency = 0

        encode_total = sum(layer_stats["latency_ms"] for layer_stats in batch_data["encode"]) # 每层的延迟加和
        decode_total = 0

        for layer in batch_data["encode"]:
            total_inter_gpu_latency += layer["latency_inter_gpu_ms"]
            total_inter_node_latency += layer["latency_inter_node_ms"]

        for step_stats in batch_data["decode"]:
            #decode_total += sum(layer_stats["latency_ms"] for layer_stats in step_stats)   # 每个时间步所有层延迟的加和
            for layer in step_stats:
                decode_total += layer["latency_ms"]
                total_inter_gpu_latency += layer["latency_inter_gpu_ms"]
                total_inter_node_latency += layer["latency_inter_node_ms"]

        summary[batch_key] = {
            "encoder_total_ms": encode_total,
            "decoder_total_ms": decode_total,
            "total_latency_ms": encode_total + decode_total,
            "total_inter_gpu_latency_ms": total_inter_gpu_latency,
            "total_inter_node_latency_ms": total_inter_node_latency
        }

    # summary["total_inter_gpu_latency_ms"] = total_inter_gpu_latency
    # summary["total_inter_node_latency_ms"] = total_inter_node_latency

    filename = os.path.join(result_dir, f"latency_summary_per_batch_{num_of_prompts}_{type}.json")
    with open (filename, "w") as f:
        json.dump(summary ,f,indent=2)
    return summary


def calculate_2round_latency(num_of_prompts,latency_summary,type):
    total_latency_per_batch = []
    inter_gpu_latency_per_batch = []
    inter_node_latency_per_batch = []

    for batch_key,batch_data in latency_summary.items():
        if not batch_key.startswith("batch_"):  # 跳过额外统计字段
            continue
        total_latency_per_batch.append(batch_data["total_latency_ms"])
        inter_gpu_latency_per_batch.append(batch_data["total_inter_gpu_latency_ms"])
        inter_node_latency_per_batch.append(batch_data["total_inter_node_latency_ms"])

    # 前4个batch最大值
    max_0_3 = max(total_latency_per_batch[:4])
    max_4_7 = max(total_latency_per_batch[4:8])

    # inter_gpu_0_3 = inter_gpu_latency_per_batch[:4]
    # inter_gpu_4_7 = inter_gpu_latency_per_batch[4:8]

    # inter_node_0_3 = inter_node_latency_per_batch[:4]
    # inter_node_4_7 = inter_node_latency_per_batch[4:8]

    idx_max_0_3 = total_latency_per_batch.index(max_0_3)
    idx_max_4_7 = total_latency_per_batch.index(max_4_7)
    # print(idx_max_0_3)
    # print(idx_max_4_7)

    total_inter_gpu_latency = inter_gpu_latency_per_batch[idx_max_0_3] + inter_gpu_latency_per_batch[idx_max_4_7]
    total_inter_node_latency = inter_node_latency_per_batch[idx_max_0_3] + inter_node_latency_per_batch[idx_max_4_7]

    result = {
        "max_latency_batch_0_3": max_0_3,
        "max_latency_batch_4_7": max_4_7,
        "communication_latency_ms": max_0_3 + max_4_7,
        "total_inter_gpu_latency_ms": total_inter_gpu_latency,
        "total_inter_node_latency_ms": total_inter_node_latency
    }

    filename = os.path.join(result_dir, f"latency_communication_{num_of_prompts}_{type}.json")
    with open (filename, "w") as f:
        json.dump(result ,f,indent=2)
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

        exflow_encode_placement = np.load(f"../different_tasks/expert_placement/{model_name}/{input_name}/{'use_bipart' if use_bipart else 'not_use_bipart'}/encode/{num}_front/intra{num_of_gpus_pre_node}_inter{num_of_nodes}.npy")
        exflow_decode_placement = np.load(f"../different_tasks/expert_placement/{model_name}/{input_name}/{'use_bipart' if use_bipart else 'not_use_bipart'}/decode/{num}_front/intra{num_of_gpus_pre_node}_inter{num_of_nodes}.npy")
        exflow_placement = np.concatenate((exflow_encode_placement, exflow_decode_placement), axis=0)

        vanilla_stats_per_batch = calculate_latency(num, routing_trace,vanilla_placement,"vanilla")
        exflow_stats_per_batch = calculate_latency(num, routing_trace, exflow_placement,"exflow")

        vanilla_summary = summarize_latency(num, vanilla_stats_per_batch,"vanilla")
        exflow_summary = summarize_latency(num, exflow_stats_per_batch,"exflow")

        vanilla_latency = calculate_2round_latency(num, vanilla_summary,"vanilla")
        exflow_latency = calculate_2round_latency(num, exflow_summary,"exflow")
