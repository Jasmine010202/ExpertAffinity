import numpy as np
from gurobipy import Model, GRB, quicksum
import csv
import random
import math
from copy import deepcopy
import os

# 成本计算
def calculate_cost(message, solution, intra_gpus=0): # message ：一个token在每一层的路由信息
    cost = 0    # 总
    count = 0   # 统计遍历的路由数
    intra = 0   #节点内
    inter = 0   #节点间
    for layer in range(len(message)-1): # 计算相邻层专家之间的通信代价
        experts_curr_layer = message[layer]  # 当前层的 top-k 专家
        experts_next_layer = message[layer + 1]  # 下一层的 top-k 专家
        for e_curr in experts_curr_layer:
            for e_next in experts_next_layer:
                if intra_gpus == 0: #节点内GPU数量为0 ，不区分节点内和节点间
                    if e_curr in solution.keys() and e_next in solution.keys():
                        count += 1
                        cost += abs(solution[e_curr] - solution[e_next])  #直接计算相邻专家message[i] 和 message[i+1] 所在 GPU 的编号差值作为通信代价
                else:   #区分节点内和节点间
                    count += 1
                    node_a = solution[e_curr] // intra_gpus # 计算所在的节点编号
                    node_b = solution[e_next] // intra_gpus
                    if solution[e_curr] != solution[e_next]: # 如果在不同的GPU上面
                        if node_a == node_b:    # 位于同一个节点
                            cost += 1
                            intra += 1
                        else:       # 位于不同节点
                            cost += 2
                            inter += 1

    if intra_gpus != 0:
        return cost / (count + 1e-8), intra / (count + 1e-8), inter / (count + 1e-8) # 返回平均 总/ 节点内/ 节点间代价
    else:
        return cost / (count + 1e-8)

'''
    for i in range(len(message)-1): # 计算相邻层专家之间的通信代价
        if intra_gpus == 0: #节点内GPU数量为0 ，不区分节点内和节点间
            if message[i] in solution.keys() and message[i+1] in solution.keys():
                count += 1
                cost += abs(solution[message[i]] - solution[message[i+1]])  #直接计算相邻专家message[i] 和 message[i+1] 所在 GPU 的编号差值作为通信代价
        else:   #区分节点内和节点间
            count += 1
            node_a = solution[message[i]] // intra_gpus # 计算所在的节点编号
            node_b = solution[message[i+1]] // intra_gpus
            if solution[message[i]] != solution[message[i+1]]: # 如果在不同的GPU上面
                if node_a == node_b:    # 位于同一个节点
                    cost += 1
                    intra += 1
                else:       # 位于不同节点
                    cost += 2
                    inter += 1
'''

def solve_graph_optimization(top_k, incremental_amount, num_layer, num_expert_per_layer, num_tokens, routing_array, 
                             fused_obj=False, cur_partition_experts=None, cur_partition_balance_experts_per_layer=0, 
                             iters=0, time_limit=60, hard_limit=0.0, partitions=2):
    # 核心目标：将 cur_partition_experts 分配到 partitions 个计算节点上
    # 使得：通信代价最小（同一节点上的专家间通信较快）。负载均衡（每个计算节点上的专家数量尽量均衡）
    # fused_obj:是否使用合并目标函数（融合通信和负载均衡）。如果 True，则目标函数同时优化通信成本和负载均衡；如果 False，则仅优化通信成本。
    # cur_partition_experts：当前参与优化的专家集合，表示本次优化考虑的专家编号
    # cur_partition_balance_experts_per_layer:当前分区内每层的专家数量均衡目标值
    # iters：优化迭代次数，控制优化过程的计算轮数
    # time_limit：控制 Gurobi 优化器的时间限制（单位：秒）
    # hard_limit：是否严格执行负载均衡约束。
    #             如果 0.0，表示负载均衡是一个软约束，优化器尽量满足；
    #             如果 >0，表示严格要求每个分区的专家数量大于等于 cur_partition_balance_experts_per_layer * hard_limit
    # partitions：计算节点的数量（或分区数）
    
    increment = incremental_amount  # 设定每次优化增加的 token 数量
    solution_storage = []   # 存储每次迭代的最优解

    layered_experts_dict = {}   # 用于按层存储专家编号
    for i in range(num_layer):
        layered_experts_dict[i] = []    

    # 计算每个专家属于哪一层，将其分配到layered_experts_dict对应层
    for k in cur_partition_experts:     
        layered_experts_dict[int(k) // num_expert_per_layer].append(k)

    # 逐步增加token迭代优化
    # start:increment起始处理量; stop:int(iters*increment)+1最大理论处理量; step:increment增量步长 
    for i in range(increment, int(iters*increment) + 1, increment):
        i = num_tokens if i > num_tokens else i
        subset_messages = routing_array[: i]        # 取前i个token的路由信息

        m = Model()                         # 创建 Gurobi 约束优化模型
        m.Params.TimeLimit = time_limit     # 设定最大运行时间
        m.setParam('Heuristics', 1.0)       # 允许 Gurobi 使用启发式算法加速求解

        # x[n, c]：二进制变量，表示专家n是否分配给计算节点c。1：专家n分配给c; 0：不分配
        x = {}  
        for n in cur_partition_experts:
            for c in range(partitions):
                x[n, c] = m.addVar(vtype=GRB.BINARY, name=f'x_{n}_{c}')
        
        # cost[k, s]：token k在s层和s+1层之间的通信代价
        # cost = {}   
        # for k in range(i):
        #     for s in range(num_layer - 1):
        #         cost[k, s] = m.addVar(vtype=GRB.BINARY, name=f'cost_{k}_{s}')
        cost = {}   
        for t in range(i):
            for s in range(num_layer - 1):
                for k1 in range(top_k):  # 当前层的 top-k
                    for k2 in range(top_k):  # 下一层的 top-k
                        cost[t, s, k1, k2] = m.addVar(vtype=GRB.BINARY, name=f'cost_{t}_{s}_{k1}_{k2}')
                

        # load_balance[layer_idx, c]：计算layer_idx层在计算节点c上的专家数相当于均衡值cur_partition_balance_experts_per_layer的偏差
        load_balance = {}   
        for layer_idx in range(num_layer):
            for c in range(partitions):
                load_balance[layer_idx, c] = m.addVar(lb=-cur_partition_balance_experts_per_layer//2, 
                                                      ub=cur_partition_balance_experts_per_layer//2+1, 
                                                      vtype=GRB.INTEGER, name=f'loadBalanceValue_{layer_idx}_{c}')

        # expert_per_layer_per_node_abs[l, n]存储每层l在每个计算节点n上的专家数偏差的绝对值变量
        expert_per_layer_per_node_abs = {}
        for l in range(num_layer):
            for n in range(partitions):
                expert_per_layer_per_node_abs[l, n] = m.addVar(lb=0, vtype=GRB.INTEGER, name=f"abs_{l}_{n}")

        # 计算每层layer_idx在计算节点c上的专家数量，并确保其偏离均衡值cur_partition_balance_experts_per_layer的差值存储在load_balance[layer_idx, c]
        # 负载偏差 = 实际专家数 - 目标专家数
        m.update()
        for layer_idx in range(num_layer):
            for c in range(partitions):
                m.addConstr(load_balance[layer_idx, c] == 
                            (sum(x[n, c] for n in layered_experts_dict[layer_idx]) - cur_partition_balance_experts_per_layer), 
                            name=f"loadConstr_{layer_idx}_{c}")

        # expert_per_layer_per_node_abs[l, n] 存储load_balance[l, n]的绝对值
        m.update()
        for l in range(num_layer):
            for n in range(partitions):
                m.addGenConstrAbs(expert_per_layer_per_node_abs[l, n], load_balance[l, n], name=f"absConstr_{l}_{n}")

        # 不使用合并目标函数，只考虑通信优化
        if not fused_obj:
            m.update()
            for layer_idx in range(num_layer):
                for c in range(partitions):
                    if hard_limit != 0.0:
                        # 允许一定范围的偏差，但专家数不能低于hard_limit设定的下限
                        m.addConstr(
                            quicksum(x[n, c] for n in layered_experts_dict[layer_idx]) >= math.ceil(cur_partition_balance_experts_per_layer * hard_limit), 
                            name=f"hardLoadBalance_{layer_idx}_{c}")
                    else:
                        # 必须严格满足均衡要求，每个计算节点每一层必须有cur_partition_balance_experts_per_layer个专家
                        m.addConstr(
                            quicksum(x[n, c] for n in layered_experts_dict[layer_idx]) == cur_partition_balance_experts_per_layer, 
                            name=f"hardLoadBalance_{layer_idx}_{c}")

        # 确保每个专家n只能被分配到一个计算节点c
        m.update()
        # m.addConstr(x[cur_partition_experts[0], 0] == 1)
        for n in cur_partition_experts:
            m.addConstr(quicksum(x[n, c] for c in range(partitions)) == 1, name=f"allone_{n}")


        # 确保所有计算节点c拥有均等的专家数,每个计算节点的专家总数必须等于cur_partition_balance_experts_per_layer * num_layer
        if hard_limit != 0.0:
            # Each partition has P/N nodes
            for c in range(partitions):
                m.addConstr(quicksum(x[n, c] for n in cur_partition_experts) == int(cur_partition_balance_experts_per_layer * num_layer), 
                            name=f"allequal_{c}")

        # print("######################################################")
        # print(f"subset_messages length: {len(subset_messages)}")
        # print(f"First 5 messages: {subset_messages[:5]}")
        # print("######################################################")

        # count_valid_step = 0    # 参与计算通信成本的token数量
        # for k in range(i):
        #     for s in range(num_layer - 1):
        #         #print("count_valid_step", count_valid_step)
        #         # 如果 subset_messages[k][s] 和 subset_messages[k][s+1] 分配在同一个分区，则 cost[k, s] = 0（无额外通信）
        #         if (subset_messages[k][s] in cur_partition_experts) and (subset_messages[k][s + 1] in cur_partition_experts):
        #             # 检查相邻层专家是否在当前计算分区
        #             count_valid_step += 1 # 在当前分区，增加 count_valid_step 计数
        #             # 添加通信成本约束
        #             for c in range(partitions):
        #                 # 如果两个专家 s 和 s+1 在同一个分区，那么 x[...] - x[...] = 0，所以 cost[k, s] 至少是 0（不会产生通信）
        #                 # 如果它们在不同的分区，那么 x[...] - x[...] = ±1，所以 cost[k, s] 至少是 1（产生跨分区通信）。
        #                 m.addConstr(cost[k, s] >= x[subset_messages[k][s], c] - x[subset_messages[k][s + 1], c], name=f"costConstr1_{k}_{s}")
        #                 # 上条约束的对称版本，确保 cost[k, s] 计算的是绝对差值，而不是负数
        #                 m.addConstr(cost[k, s] >= x[subset_messages[k][s + 1], c] - x[subset_messages[k][s], c], name=f"costConstr2_{k}_{s}")

        count_valid_step = 0    # 参与计算通信成本的token数量
        for k in range(i):
            for s in range(num_layer - 1):
                for k1 in range(top_k):
                    for k2 in range(top_k):
                        e_curr = subset_messages[k][s][k1]
                        e_next = subset_messages[k][s + 1][k2]
                        if e_curr in cur_partition_experts and e_next in cur_partition_experts:
                            # 检查相邻层专家是否在当前计算分区
                            count_valid_step += 1 # 在当前分区，增加 count_valid_step 计数
                            # 添加通信成本约束
                            for c in range(partitions):
                                # 如果两个专家 s 和 s+1 在同一个分区，那么 x[...] - x[...] = 0，所以 cost[k, s] 至少是 0（不会产生通信）
                                # 如果它们在不同的分区，那么 x[...] - x[...] = ±1，所以 cost[k, s] 至少是 1（产生跨分区通信）。
                                m.addConstr(cost[t, s, k1, k2] >= x[e_curr, c] - x[e_next, c], name=f"costConstr1_{t}_{s}_{k1}_{k2}")
                                m.addConstr(cost[t, s, k1, k2] >= x[e_next, c] - x[e_curr, c], name=f"costConstr2_{t}_{s}_{k1}_{k2}")

        # Objective
        # 控制优化目标
        if fused_obj:
            # 同时优化通信成本和负载均衡
            # print("fused_obj", count_valid_step)
            m.setObjective(quicksum(cost[t, s, k1, k2] for t in range(i) for s in range(num_layer - 1) for k1 in range(top_k) for k2 in range(top_k)) / count_valid_step
                            + 1 / num_layer / cur_partition_balance_experts_per_layer 
                            * quicksum(expert_per_layer_per_node_abs[l, n] for l in range(num_layer) for n in range(partitions)), GRB.MINIMIZE)
        else:
            # 只关注通信成本
            # print("not fused_obj", count_valid_step)
            # m.setObjective(quicksum(cost[k, s] for k in range(i) for s in range(num_layer - 1)) / count_valid_step, GRB.MINIMIZE)
            m.setObjective(quicksum(cost[t, s, k1, k2] for t in range(i) for s in range(num_layer - 1) for k1 in range(top_k) for k2 in range(top_k)) / count_valid_step, 
                           GRB.MINIMIZE)

        # 同步模型中的所有更改，确保添加的变量、约束和目标函数正确生效
        m.update()

        # 如果i足够大（即 token 数量足够多），则加载上一次的求解结果
        # Gurobi 允许设置变量 start，这样求解器可以从已有解开始搜索，加速收敛
        if i//increment > 1:
            for n, c in solution_storage[-1].items():
                m.getVarByName(f"x_{n}_{c}").start = 1.0
            print("Loading solutions...")

        # 调用 Gurobi 优化器，求解目标函数的最优解。
        m.optimize()

        if m.SolCount > 0:
            # Gurobi 找到了一个可行解
            solution = {}
            for n in cur_partition_experts:
                for c in range(partitions):
                    if x[n, c].x > 0.5: # x[n, c]> 0.5，专家n被分配到c，把solution[n]=c记录到solution
                        solution[n] = c # solution 存储每个专家n所分配的计算节点c
            print(f"Complete {i/num_tokens*100}%: {solution}")
            solution_storage.append(solution)

            # 计算message经过当前solution方案的通信成本 并累加，再取平均
            avg_cost = sum(calculate_cost(message, solution) for message in routing_array) / num_tokens
            load_balance_output = {}
            for layer_idx in range(num_layer):
                for c in range(partitions):
                    load_balance_output[layer_idx, c] = load_balance[layer_idx, c].x    # 获取 load_balance[layer_idx, c] 变量的最优值

            print(f"Complete {i/num_tokens*100}%, Average cost per token: {avg_cost}, \
                  load balance max: {np.max(np.array(list(load_balance_output.values())))}, \
                  load balance min: {np.min(np.array(list(load_balance_output.values())))}, \
                  load balance stdv: {np.std(np.array(list(load_balance_output.values())))}")
        else:
            # 找不到可行解
            print("No solution found.")
        del m
    # if not solution_storage:
    #     return None
    
    return solution_storage[-1] # 最后一个求解结果，即最优专家分配方案

# 按照平均分配策略的专家放置结果
def vanilla_placement(num_layer, num_expert_per_layer, intra_gpus, nodes):
    overall_gpus = intra_gpus * nodes   # 总GPU数量
    expert_per_gpu = num_expert_per_layer / overall_gpus    # 每个GPU每层容纳多少个专家
    placement = {}
    for i in range(num_layer * num_expert_per_layer): # 对于所有专家
        # i // expert_per_gpu 计算该专家属于哪一组 GPU
        # % overall_gpus确保GPU号在 [0, overall_gpus-1] 之间循环分配
        placement[i] = int((i // expert_per_gpu) % overall_gpus)    
    return placement

def read_parition(top_k, file_dir, result_dir, routing_array, num_tokens, num_layer, num_expert_per_layer,  intra_gpus, nodes, incremental_amount, run_times, time_limits,total_number_gpu, use_bipart=False):
    # incremental_amount：用于优化的增量参数。
    # run_times 和 time_limits：优化算法的迭代次数和时间限制。
    solution_dict = {}
    cur_partition_experts = [i for i in range(num_layer * num_expert_per_layer)]

    # 二分优化策略
    if use_bipart:
        cur_partition_balance_experts_per_layer = num_expert_per_layer // 2 # 初始每分区专家数为每层专家数的一半
        total_level = int(np.log2(total_number_gpu)) # 分区数？
        for i in range(total_level):
            if i == 0: # 第一轮分区，先把所有专家分成两个大区
                cur_solution = solve_graph_optimization(top_k, incremental_amount, num_layer, num_expert_per_layer, num_tokens, routing_array, False, cur_partition_experts, cur_partition_balance_experts_per_layer, iters=run_times, time_limit=time_limits, partitions=2)
                upper_level_solution = deepcopy(cur_solution) # 保存上一轮的专家分配方案，用于进一步优化
            else:   # 之后每一轮进一步细分先前的分区
                num_problems = 2 ** i # 每一轮将上一轮的2^i个分区再拆分成更小的部分
                for j in range(num_problems): # 遍历当前2^i 个已分区的组，每个组再进行更细粒度优化
                    cur_partition_experts = []
                    # 把上一轮分区j的专家挑出来放进 cur_partition_experts
                    for k, v in upper_level_solution.items(): 
                        if v == j:
                            cur_partition_experts.append(k)
                    # 让 cur_partition_experts 在 j 号分区内部 再细分成 2 组
                    sub_solution = solve_graph_optimization(top_k, incremental_amount, num_layer, num_expert_per_layer, num_tokens, routing_array,False, cur_partition_experts, cur_partition_balance_experts_per_layer, iters=run_times, time_limit=time_limits, partitions=2)
                    for k, v in sub_solution.items():
                        # cur_solution[k]：当前专家 k 在上一层分配的分区号，上层分区号 × 2 + 本层分区号
                        cur_solution[k] = cur_solution[k] * 2 + v
                upper_level_solution = deepcopy(cur_solution) # 逐步存储每一层的优化方案

            cur_partition_balance_experts_per_layer = cur_partition_balance_experts_per_layer // 2

        cur_solution = upper_level_solution

    # 非二分优化策略，先节点间，后节点内
    else:
        cur_partition_balance_experts_per_layer = num_expert_per_layer // nodes # 计算每个计算节点内的专家数量
        print(len(cur_partition_experts))

        # 节点数超过1，先跨节点分配
        if nodes > 1:
            # partitions=nodes 跨节点专家划分
            cur_solution = solve_graph_optimization(top_k, incremental_amount, num_layer, num_expert_per_layer, num_tokens, routing_array, False, cur_partition_experts, cur_partition_balance_experts_per_layer, iters=run_times, time_limit=time_limits, partitions=nodes)
            upper_level_solution = deepcopy(cur_solution)
            cur_partition_balance_experts_per_layer = cur_partition_balance_experts_per_layer // intra_gpus # 计算每个GPU 内部的专家数量

            for i in range(nodes): # 遍历所有节点，在每个节点内部分配专家到节点内的GPU上
                cur_partition_experts = []
                partition_offset = i * intra_gpus
                for k, v in upper_level_solution.items(): # 筛选所有属于这个节点的专家
                    if v == i:
                        cur_partition_experts.append(k)
                print(len(cur_partition_experts))
                # 在节点内部进一步优化专家到节点内的GPU上
                sub_solution = solve_graph_optimization(top_k, incremental_amount, num_layer, num_expert_per_layer, num_tokens, routing_array, False, cur_partition_experts, cur_partition_balance_experts_per_layer, iters=run_times, time_limit=time_limits, partitions=intra_gpus)
                for k, v in sub_solution.items():
                    cur_solution[k] = v + partition_offset
        
        # 节点数为1，直接跨GPU分配
        else:
            cur_partition_balance_experts_per_layer = cur_partition_balance_experts_per_layer // intra_gpus
            cur_solution = solve_graph_optimization(top_k, incremental_amount, num_layer, num_expert_per_layer, num_tokens, routing_array, False, cur_partition_experts, cur_partition_balance_experts_per_layer, iters=run_times, time_limit=time_limits, partitions=intra_gpus)
    
    ######################
    vanilla_p = vanilla_placement(num_layer, num_expert_per_layer, intra_gpus, nodes)
    print(vanilla_p)
    print("Vanilla placement:")
    cost = 0
    intra = 0
    inter = 0
    for message in routing_array:   # 遍历每一条路由信息
        cost_, intra_, inter_ = calculate_cost(message, vanilla_p, intra_gpus=intra_gpus) #总通信代价、节点内通信代价、节点间通信代价
        cost += cost_
        intra += intra_
        inter += inter_
    avg_cost = cost / num_tokens * (num_layer - 1) #每个token的跨层通信代价
    avg_intra = intra / num_tokens * (num_layer - 1) / (intra_gpus - 1) # 每个token跨层通信在每对GPU之间的通信代价
    if nodes > 1:
        avg_inter = inter / num_tokens * (num_layer - 1) / (intra_gpus * (nodes - 1))
    else:
        avg_inter = 0
        
    print(avg_cost, avg_intra, avg_inter)

    #######################
    print(cur_solution)
    cost = 0
    intra = 0
    inter = 0
    for message in routing_array:
        cost_, intra_, inter_ = calculate_cost(message, cur_solution, intra_gpus=intra_gpus)
        cost += cost_
        intra += intra_
        inter += inter_
    avg_cost = cost / num_tokens * (num_layer - 1)
    avg_intra = intra / num_tokens * (num_layer - 1) / (intra_gpus - 1)
    if nodes > 1:
        avg_inter = inter / num_tokens * (num_layer - 1) / (intra_gpus * (nodes - 1))
    else:
        avg_inter = 0

    file_name = f'{file_dir}/solution_intra{intra_gpus}_inter{nodes}_cost{avg_cost}_cintra{avg_intra}_cinter{avg_inter}.csv'

    with open(os.path.join(file_name), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Expert", "Node"])
        for expert, node in cur_solution.items():
            writer.writerow([expert, node])

    print("After optimization:")
    print(avg_cost, avg_intra, avg_inter)

    ################################################
    # 恢复专家ID 行：每一层 列：每一层的专家ID 值：GPU ID
    expert_placement = np.zeros((num_layer, num_expert_per_layer), dtype=int)
    for expert, node in cur_solution.items():
        layer = expert // num_expert_per_layer  # 计算层号
        original_expert_id = expert % num_expert_per_layer  # 计算原始专家ID
        expert_placement[layer, original_expert_id] = node  # 填充节点编号
    
    print("Expert Placement (Layer x Expert):")
    print(expert_placement)
    
    np.save(f"{result_dir}/intra{intra_gpus}_inter{nodes}.npy", expert_placement)

    # 均匀放置的方案： 2节点每个4个GPU，一层8个专家
    # average_placement = np.zeros((num_layer, num_expert_per_layer), dtype=int)
    # for expert, node in vanilla_p.items():
    #     layer = expert // num_expert_per_layer  # 计算层号
    #     original_expert_id = expert % num_expert_per_layer  # 计算原始专家ID
    #     average_placement[layer, original_expert_id] = node  # 填充节点编号
    
    # print("Vanilla Placement (Layer x Expert):")
    # print(average_placement)
    # np.save(f"{result_dir}/intra{intra_gpus}_inter{nodes}_vanilla.npy", average_placement)

    # # 保存CSV 文件
    # result_file_name = f'expert_placement/{model_name}/{input_name}/{phrase_mode}/intra{intra_gpus}_inter{nodes}.csv'
    # with open(os.path.join(result_file_name), 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['Layer', 'Expert', 'Node'])
    #     for layer in range(num_layer):
    #         for expert in range(num_expert_per_layer):
    #             writer.writerow([layer, expert, expert_placement[layer, expert]])
    ################################################


def placement_plan(routing_array, model_name, input_name, phrase_mode, prompt_num, top_k):
    #routing_array = np.load(f'expert_trace/{model_name}/{input_name}/{phrase_mode}_routing_trace.npy') # [num_tokens, num_MOE_layers], expert id for each token at each layer
    num_tokens, _, _= routing_array.shape
    print(num_tokens)
    num_layer = routing_array.shape[1]
    num_expert_per_layer = 64 # 每层的专家数
    total_experts = num_expert_per_layer * num_layer 
    assert total_experts % 2 == 0
    intra_gpus = 2 #8 4 节点内的GPU数量
    nodes = 2 #1 节点数量
    use_bipart = True # True False
    incremental_amount = 5000  #5000
    run_times = (num_tokens + incremental_amount - 1) // incremental_amount
    time_limits = 60 # By default, we use bipart solver. If not, you can increase the search time accordingly.

    #top_k = 8 # 2 4 8 16

    file_dir = f"placement_results/{model_name}/{input_name}/{'use_bipart' if use_bipart else 'not_use_bipart'}/{phrase_mode}/top{top_k}/{prompt_num}"
    os.makedirs(file_dir, exist_ok=True)

    result_dir = f"expert_placement/{model_name}/{input_name}/{'use_bipart' if use_bipart else 'not_use_bipart'}/{phrase_mode}/top{top_k}/{prompt_num}"
    os.makedirs(result_dir, exist_ok=True)

    for i in range(num_layer):
        routing_array[:, i] += num_expert_per_layer * i # 专家编码唯一
    print(routing_array)

    # print("######################################################")
    # print("routing_array shape:", routing_array.shape)
    # print("First 5 rows:", routing_array[:5])
    # print(f"num_tokens: {num_tokens}")
    # print(f"num_layer: {num_layer}")
    # print("######################################################")

    read_parition(top_k, file_dir, result_dir, routing_array, num_tokens, num_layer, num_expert_per_layer, intra_gpus, nodes, incremental_amount, run_times, time_limits, intra_gpus * nodes, use_bipart)
