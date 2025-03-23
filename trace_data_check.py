import numpy as np
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



# trace_data_st = np.load(f"expert_trace/Switch_Transformer/5_input/decode_routing_trace.npy")
# print(trace_data_st.shape)
# print(trace_data_st[:5])

# trace_data_ol = np.load(f"expert_trace/OLMoE/gigaword/top8/decode_routing_trace_8.npy")
# print(trace_data_ol.shape)
# print(trace_data_ol[:5])

expert_placement = np.load(f"expert_placement/OLMoE/gigaword/use_bipart/decode/top8/32/intra2_inter2.npy")
print(expert_placement.shape)
print(expert_placement)