import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import time
import torch
import random
import json
from transformers import AutoTokenizer
# from src.modeling_olmoe_parallel import OlmoeForCausalLM # 平均放置
from src.modeling_olmoe_placement import OlmoeForCausalLM # 按层内亲和性放置
# from src.modeling_olmoe import OlmoeForCausalLM # 原本，1张卡
# from src.test import OlmoeForCausalLM # 平均放置
import torch.distributed as dist
from accelerate import Accelerator


def main():
    accelerator = Accelerator()
    device = accelerator.device

    #device = torch.device("cuda:0")
    #device = "cuda" if torch.cuda.is_available() else "cpu"

    print("🚀 当前进程信息：")
    print(f"  ↪ accelerator.process_index = {accelerator.process_index}")
    print(f"  ↪ device = {accelerator.device}")
    print(f"  ↪ local_process_index = {accelerator.local_process_index}")
    print(f"  ↪ total number of processes (GPUs) = {accelerator.num_processes}")
    print(f"  ↪ torch.cuda.device_count() = {torch.cuda.device_count()}")

    tokenizer = AutoTokenizer.from_pretrained("./models/OLMoE-1B-7B-0924")
    tokenizer.padding_side = "left"

    # average/balance/affinity
    model = OlmoeForCausalLM.from_pretrained("./models/OLMoE-1B-7B-0924", torch_dtype=torch.float16)

    device_main = torch.device("cuda:0")
    model.lm_head.to(device_main)
    model.model.embed_tokens.to(device_main)
    model.model.norm.to(device_main)
    model.model.rotary_emb.to(device_main)
    for layer in model.model.layers:
        layer.input_layernorm.to(device_main)
        layer.post_attention_layernorm.to(device_main)
        layer.self_attn.to(device_main)
        layer.mlp.gate.to(device_main)  


    print("💡 非专家部分所在设备：")
    print("Embedding:", model.model.embed_tokens.weight.device)
    print("Attention layer 0:", next(model.model.layers[0].self_attn.parameters()).device)
    print("LM Head:", model.lm_head.weight.device)


    # 1gpu
    # model = OlmoeForCausalLM.from_pretrained("./models/OLMoE-1B-7B-0924", torch_dtype=torch.float16).to(device)

    model.eval()

    # for i, expert in enumerate(model.moe.experts):
    #     print(f"Expert {i} is on device: {next(expert.parameters()).device}")

    # for name, param in model.named_parameters():
    #     print(f"{name} on {param.device}")
    #     break  # 只看一个


    prompt_sizes = [8, 16, 32, 64, 128, 256, 512, 1024] # 8, 16, 32, 64, 128, 256, 512, 1024
    
    batch_size = 32

    output_dir = "./test_affinity_debug/acc_1process_4gpus/balance_2_new"
    os.makedirs(output_dir, exist_ok=True)

    all_inference_time = {}

    for size in prompt_sizes:
        print(f"\nPrompt size: {size}")
        # 加载 prompts
        with open(f"./dataset/sonnet/sonnet_{size}.txt", "r") as file:
            prompts = [line.strip() for line in file.readlines() if line.strip()]

        #sampled_prompts = random.sample(prompts, size)

        start_time = time.time()
        decoded_outputs = []

        for i in range(0, size, batch_size):
            #batch_prompts = sampled_prompts[i:i + batch_size]
            batch_prompts = prompts[i:i+batch_size]
            # average
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            #inputs = accelerator.prepare(inputs) 
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # #affinity
            # inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            # inputs = accelerator.prepare(inputs) 
            # t1 = time.time()
            # inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            # inputs = {k: v.to(device) for k, v in inputs.items()}
            # print("Tokenization time:", time.time() - t1)

            # 1 gpu
            # inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)


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
            #for prompt, result in zip(sampled_prompts, decoded_outputs):
            for prompt, result in zip(prompts, decoded_outputs):
                output_file.write(f"Prompt: {prompt}\nGenerated: {result}\n\n")

        print(f"End of inference. ({size} prompts)")

    # 保存时间记录
    timing_file_path = os.path.join(output_dir, "all_inference_time.json")
    with open(timing_file_path, "w") as f:
        json.dump(all_inference_time, f, indent=4)

    print(f"All inference time saved to : {timing_file_path}")

    if dist.is_initialized() and accelerator.distributed_type != "NO":
        dist.destroy_process_group()


if __name__ == "__main__":
    main()