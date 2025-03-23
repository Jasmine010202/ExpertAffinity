from transformers import AutoTokenizer
from src.modeling_switch_transformers import SwitchTransformersForConditionalGeneration
#from plot import plot_expert_selection_distribution as draw

import os
import numpy as np
import random 

tokenizer = AutoTokenizer.from_pretrained("./models/switch-base-8")
model = SwitchTransformersForConditionalGeneration.from_pretrained("./models/switch-base-8")

with open("./dataset/gigaword_test.txt", "r") as file:
    prompts = [line.strip() for line in file.readlines() if line.strip()]

prompt_sizes = [8, 16, 32, 64, 128, 256, 512, 1024]

output_dir = "./output/gigaword"
os.makedirs(output_dir, exist_ok=True)

routing_data_dir = "./expert_trace/Switch_Transformer/gigaword"
os.makedirs(routing_data_dir, exist_ok=True)

for size in prompt_sizes:
    #print(size)
    #print("Encode\n",model.encode_routing_trace)
    #print("Decode\n",model.decode_routing_trace)

    sampled_prompts = random.sample(prompts, size)
    output_file_path =  f"{output_dir}/output_{size}.txt"
    
    model.encode_routing_trace = []  
    model.decode_routing_trace = []

    # print("After Clear")
    # print("Encode",model.encode_routing_trace)
    # print("Decode",model.decode_routing_trace)

    with open(output_file_path, "w") as output_file:
        for prompt in sampled_prompts:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids  # Batch size 1
            outputs = model.generate(input_ids, max_new_tokens=50)
            decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            output_file.write(f"Prompt: {prompt}\nGenerated Summary: {decoded_text}\n")
    
    print(f"End of inference.({size} prompts)")

    # 保存路由信息
    with open(output_file_path, "a") as output_file:
        output_file.write(f"encode:\n{model.encode_routing_trace}\ndecode:\n{model.decode_routing_trace}\n")

    if model.encode_routing_trace:
        encode_trace = np.concatenate(model.encode_routing_trace, axis=0)
        np.save(f"{routing_data_dir}/encode_routing_trace_{size}.npy", encode_trace)
        print(f"Saved: {routing_data_dir}/encode_routing_trace_{size}.npy")

    if model.decode_routing_trace:
        decode_trace = np.concatenate(model.decode_routing_trace, axis=0)
        np.save(f"{routing_data_dir}/decode_routing_trace_{size}.npy", decode_trace)
        print(f"Saved: {routing_data_dir}/decode_routing_trace_{size}.npy")


# for prompt in prompts:
#     input_ids = tokenizer(prompt, return_tensors="pt").input_ids  # Batch size 1
#     outputs = model.generate(input_ids, max_new_tokens=50)
#     decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     with open(f"{output_dir}/output.txt", "a") as output_file:
#         output_file.write(f"Prompt: {prompt}\nGenerated Summary: {decoded_text}\n")

# print("End of inference")        

# with open(f"{output_dir}/output.txt", "a") as output_file:
#     output_file.write(f"encode:\n{model.encode_routing_trace}\ndecode:\n{model.decode_routing_trace}\n")


# if model.encode_routing_trace:
#     encode_trace = np.concatenate(model.encode_routing_trace, axis=0)
#     np.save(f"{routing_data_dir}/encode_routing_trace.npy", encode_trace)
#     print(f"Saved:{routing_data_dir}/encode_routing_trace.npy")

# if model.decode_routing_trace:
#     decode_trace = np.concatenate(model.decode_routing_trace, axis=0)
#     np.save(f"{routing_data_dir}/decode_routing_trace.npy", decode_trace)
#     print(f"Saved:{routing_data_dir}/decode_routing_trace.npy")


#检查数据格式：
# encode_trace_data = np.load(f"{routing_data_dir}/encode_routing_trace.npy")
# decode_trace_data = np.load(f"{routing_data_dir}/decode_routing_trace.npy")

# print("Encode Trace Shape:", encode_trace_data.shape)  # (num_tokens, num_layers)
# print(encode_trace_data[:15])
# print("Decode Trace Shape:", decode_trace_data.shape)  # (num_tokens, num_layers)
# print(decode_trace_data[:15])


# # 改批处理
# batch_size = 8 
# with open(f"{output_dir}/output.txt", "w") as output_file:
#     for i in range(0, len(prompts), batch_size):
#         batch_prompts = prompts[i:i + batch_size]  # 获取当前 batch
#         batch_inputs = tokenizer(batch_prompts, padding=True, truncation=True, return_tensors="pt")
#         batch_outputs = model.generate(batch_inputs.input_ids, max_new_tokens=50)
#         decoded_texts = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)

#         for prompt, summary in zip(batch_prompts, decoded_texts):
#             output_file.write(f"Prompt: {prompt}\nGenerated Summary: {summary}\n\n")
