import numpy as np
import json
#检查数据格式：
# encode_trace_data = np.load("expert_trace/Switch_Transformer/5_input/encode_routing_trace.npy")
# decode_trace_data = np.load("expert_trace/Switch_Transformer/5_input/decode_routing_trace.npy")

# print("Encode Trace Shape:", encode_trace_data.shape)  # (num_tokens, num_layers)
# print(encode_trace_data[:15])
# print("Decode Trace Shape:", decode_trace_data.shape)  # (num_tokens, num_layers)
# print(decode_trace_data[:15])


model_names = ["Switch_Transformer"]
input_names = ["gigaword"]  # "5_input","50_input","100_input",
phrase_mode = ["encode","decode"]
prompt_sizes = [8, 16, 32, 64, 128, 256, 512, 1024]

# for model_name in model_names:
#     for input_name in input_names:
#         for mode in phrase_mode:
#                 for size in prompt_sizes:
#                     trace_data = np.load(f"expert_trace/{model_name}/{input_name}/{mode}_routing_trace_{size}.npy")
#                     print(f"{model_name}/{input_name}/{mode}/{size} - num of tokens:",trace_data.shape[0])
#                     print(trace_data[:5])



trace_data_st = np.load(f"/data/workspace/hanyu2/ExpertAffinity/Occult_test/expert_trace/used_for_occult/by_prompt/OLMoE_sonnet_top8/decode_routing_trace_512.npy")
print(trace_data_st.shape)
print(trace_data_st[0][0])

# trace_data = np.load(f"/data/workspace/hanyu2/ExpertAffinity/Occult_test/expert_trace/traffic_test/by_prompt/OLMoE_GSM8K_top8/decode_routing_trace_512.npy")
# print(trace_data.shape)
# print(trace_data[0][0])
# print(trace_data_st[:5])

# def load_prompt_trace_from_jsonl(prompt_id, path):
#     with open(path, "r") as f:
#         for line in f:
#             item = json.loads(line)
#             if item["prompt_id"] == prompt_id:
#                 trace = np.array(item["trace"], dtype=int)
#                 return trace
#     raise ValueError(f"Prompt ID {prompt_id} not found.")


# trace = load_prompt_trace_from_jsonl(0, f"./different_tasks/trace_by_prompt/expert_trace/Switch_Transformer/generate/sonnet/routing_trace_512.jsonl")
# print(f"Prompt {0} trace shape: {trace.shape}")
# print("First 5 rows:")
# print(trace)



# trace_data_ol = np.load(f"expert_trace/OLMoE/gigaword/top8/decode_routing_trace_8.npy")
# print(trace_data_ol.shape)
# print(trace_data_ol[:5])

# expert_placement = np.load(f"expert_placement/OLMoE/gigaword/use_bipart/decode/top8/32/intra2_inter2.npy")
# print(expert_placement.shape)
# print(expert_placement)