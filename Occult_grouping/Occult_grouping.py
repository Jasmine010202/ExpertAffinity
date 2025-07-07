from typing import List

import torch

# 用于模块导出控制；指定当使用 from module import * 时，只会导入这两个函数
__all__ = ["group_experts_on_collaboration", "group_experts_on_collaboration_heterogeneous_group"]


def _check_expert_collab_counts(expert_collab_counts: torch.Tensor):
    assert expert_collab_counts.size(0) == expert_collab_counts.size(1), "Expert collaboration matrix must be square."
    # Check if the matrix is symmetric
    assert torch.allclose(expert_collab_counts,
                          expert_collab_counts.T), "Expert collaboration matrix must be symmetric."
    # Check if the diagonal is zero
    assert torch.all(expert_collab_counts.diag() == 0), "Diagonal elements must be zero."

    ###################################################################
    # # 防止陷入死循环
    # # 检查是否有专家从未被激活（行和列为0）
    # row_sums = expert_collab_counts.sum(dim=1)
    # if torch.any(row_sums == 0):
    #     inactive_experts = torch.nonzero(row_sums == 0).squeeze().tolist()
    #     raise ValueError(f"The following expert(s) have never been activated or collaborated with others: {inactive_experts}")

    # # 检查是否所有专家之间都没有协作（整个矩阵除了对角线都是0）
    # if torch.all(expert_collab_counts == 0):
    #     raise ValueError("No expert collaborations found. The entire collaboration matrix is zero.")
    
    ###################################################################


def compute_entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)



# 只能逐层算。expert_collab_counts要逐层给
def group_experts_on_collaboration(
        expert_collab_counts: torch.Tensor,
        num_groups: int,
        even_groups: bool = True,
) -> List[List[int]]:
    """
    Group experts based on the number of times they have collaborated with each other. Experts with more collaborations
    between each other will be grouped together.

    Greedy algorithm:
        1. Find the maximum value (x,y) in the matrix.
            - If none of the two experts has been grouped, group them together in the same new group;
                But if the number of groups is full, go through all the existing groups with at least two empty slots
                and find the one with the highest sum of collaboration counts with the two experts, and group them into
                that group.
            - If one of them has been grouped, group the other one into the same group;
                But if the group is full, set this element to 0 and repeat this step.

        2. Set all the rows & columns of x and y to 0 in the matrix. Repeat step 1 until all the elements in the matrix
            are 0.


    Parameters
    ----------
    expert_collab_counts: torch.Tensor
        A tensor of shape (num_experts, num_experts) where each element (i, j) represents the number of times expert i
        has collaborated with expert j.
    num_groups: int
        The number of groups to create.
    even_groups: bool
        Whether to create groups of equal sizes.

    Returns
    -------
    List[List[int]]
        A list of lists where each sublist contains the indices of the experts that should be grouped together.

    Examples
    --------
    >>> heatmap = torch.arange(16*16).reshape(16, 16) + torch.arange(16*16).reshape(16, 16).T
    >>> heatmap = heatmap.fill_diagonal_(0)
    >>> group_experts_on_collaboration(heatmap, num_groups=4, even_groups=True)
    [[14, 15, 13, 12], [10, 11, 9, 8], [6, 7, 5, 4], [2, 3, 1, 0]]

    """
    num_experts = expert_collab_counts.size(0)

    # 不支持非均匀分组
    if not even_groups:
        raise NotImplementedError("Uneven groups are not supported yet.")
    else:
        group_capacity = num_experts // num_groups

    # 检查矩阵格式是否合法
    _check_expert_collab_counts(expert_collab_counts)

    # Create a mask to keep track of the experts that have been grouped
    # 布尔向量，标记每个专家是否已经被分组
    grouped_mask = torch.zeros(num_experts, dtype=torch.bool)
    # Create a list of groups
    # 存储分组结果，每个元素是一个专家组
    group_list = []

    # 查找一个专家现在属于哪个组，还未分组就返-1
    def _find_expert_group(exp_id: int) -> int:
        for _i, _group in enumerate(group_list):
            if exp_id in _group:
                return _i
        return -1

    '''
        陷入死循环：
            1. 划分slow和fast expert的时候, compute_entropy 的结果只关注“是否广泛协作”,不在乎有没有和其他 fast experts 协作,也不管“总协作强度”,fast_experts组内可能会出现有些专家协作次数为0
            2. 或者是在协作矩阵中，本身就有一些专家从未被激活

            贪婪分组算法, 基于一对expert pair的协作关系进行贪婪分组
            分组的目标是让协作最多的专家尽量在同一组
            只在每次循环中处理当前最大值 (x, y)，然后把 expert_collab_counts[x, y] = 0

            但如果:
                所有未处理的 (x, y) 协作为 0, 无法配对，也无法加组
                每一对都不能加入任何组（因为组满或不满足条件）；
                而且这两专家仍然没有被 grouped_mask[...] = True 标记；
            就会永远重复这个过程,陷入死循环
    '''

    # 只要还有专家没有被分组，就继续执行
    while not torch.all(grouped_mask):
        # Find the maximum value in the matrix 在还没被分组的专家之间，找出协作次数最大的那一对 (x, y)
        # (N, N) * (1, N)  → 每行乘上这个布尔 mask（列方向有效，行方向没影响）
        max_idx = torch.argmax(expert_collab_counts * ~grouped_mask).item()
        #  torch.argmax() 默认在整个张量上操作，会把张量展平成一维，然后找出最大值的位置，并返回它在这个展平后的一维向量中的索引
        # 转成二维矩阵中的坐标
        # Convert the flattened index to 2D index
        x, y = max_idx // num_experts, max_idx % num_experts

        # 如果两个专家都已经分组了，把协作次数设置成0，跳过
        # Check if both experts have been grouped
        if grouped_mask[x] and grouped_mask[y]:
            # If both experts have been grouped, set the element to 0
            expert_collab_counts[x, y] = 0
            continue

        # Group the experts together

        # 如果其中一个专家已经分组，就把另一个加入该组
        if grouped_mask[x]:
            expert_idx = y
            group_idx = _find_expert_group(x)
        elif grouped_mask[y]:
            expert_idx = x
            group_idx = _find_expert_group(y)

        else: 
            # 如果两个专家都没有分组
            # If none of the experts have been grouped, try if we can group them together in a new group
            if len(group_list) < num_groups:
                # 如果还可以建立新组，加入新组（如果当前这组的协作次数是N，其中一个专家和已分组的某个专家协作次数是N-1，但是这样分在了两个组里）
                group_list.append([x, y])
                grouped_mask[x] = True
                grouped_mask[y] = True
                continue
            else:
                # If the number of groups is full, find the group with at least two empty slots
                # 所有组已满，找个最合适的组-还能容纳两个专家的，且这组与当前这两个专家协作次数最多
                group_idx = -1
                max_collab = 0
                for i, group in enumerate(group_list):
                    # 只有当该组还能容纳至少两个专家时才考虑
                    if len(group) < group_capacity - 1:
                        # Find the sum of collaboration counts with the two experts
                        collab_sum = expert_collab_counts[group, x].sum() + expert_collab_counts[group, y].sum()
                        # 找和这两个专家协作次数最多的组
                        if collab_sum > max_collab:
                            group_idx = i
                            max_collab = collab_sum

                # 如果找不到合适的组，就把协作次数置为0（这两个专家不能放到一起去，置为0避免下次还是这一对，陷入死循环）
                if group_idx == -1:
                    # None of the groups has at least two empty slots, set both elements to 0
                    expert_collab_counts[x, y] = 0
                    continue
                
                # 找到合适的组，加入，标记专家已分组
                group_list[group_idx].extend([x, y])
                grouped_mask[x] = grouped_mask[y] = True
                continue

        # 主要针对有一个专家已经被分组的情况
        assert group_idx != -1, "One of the experts should be in a group."

        # Check if the group is full
        # 检查这个组有没有满，没有满，加入；满了，协作次数置0
        if len(group_list[group_idx]) < group_capacity:
            group_list[group_idx].append(expert_idx)
            grouped_mask[expert_idx] = True
        else:
            expert_collab_counts[x, y] = 0


    if even_groups:
        # Sanity check
        # 强制校验，如果有组不满足专家数的要求，报错
        if not all(len(group) == num_experts // num_groups for group in group_list):
            print(f"Warning: {len(group_list)} groups with sizes {[len(group) for group in group_list]}")
            print(f"expert_collab_counts shape: {expert_collab_counts.shape}")
            print(f"num_groups: {num_groups}")
            print(f"expert_collab_counts: {expert_collab_counts}")
            raise RuntimeError("Uneven groups detected.")

    return group_list


def group_experts_on_collaboration_heterogeneous_group(
        expert_collab_counts: torch.Tensor,
        num_groups: int,
        num_fast_groups: int,
        even_groups: bool = True,
):
    """
    Group experts based on the number of times they have collaborated with each other. Experts with more collaborations
    between each other will be grouped together.

    Greedy algorithm:
        1. Divide all the experts into two groups: fast and slow. The fast group contains the most collaborative
            experts, and the slow group contains the rest which are prone to have pair-wise collaborations.
            【fast 组包含最活跃(协作最多)的专家: slow 组是剩下的，倾向于只和少数人合作。】

        2. For the experts in the slow group, iteratively pick pairs of experts that have the highest collaboration
            counts. Calculate the average collaboration counts between each pair of them. Place those pairs with the
            highest average collaboration counts on the same group. Repeat this process until all the experts in the
            slow group are grouped. Make sure: 1) each pair of experts are in the same group; 2) each group has the
            same number of experts.
            【对slow组中的专家,反复挑选协作次数最多的一对组成一个专家对。然后根据这些专家对之间的平均协作强度,把协作强度高的配对分进同一个组。保证:1)每对专家被分在同一个组;2)各组专家数量一致。】

        3. For the experts in the fast group, reuse the group_experts_on_collaboration function to group them. Make
            sure each group has the same number of experts.
            【对 fast 专家，直接使用已有的 `group_experts_on_collaboration` 函数来进行分组，同样保证每组专家数量一致。】

    Parameters
    ----------
    expert_collab_counts: torch.Tensor
        A tensor of shape (num_experts, num_experts) where each element (i, j) represents the number of times expert i
        has collaborated with expert j.
    num_groups: int
        The number of groups to create.
    num_fast_groups: int
        The number of fast groups to create, where the most collaborative experts are places.
    even_groups: bool
        Whether to create groups of equal sizes.

    Returns
    -------
    List[List[int]]
            A list of lists where each sublist contains the indices of the experts that should be grouped together.

    Examples
    --------
    >>> heatmap = torch.arange(16*16).reshape(16, 16) + torch.arange(16*16).reshape(16, 16).T
    >>> heatmap[[0, 1, 2, 3, 11, 12, 13, 14], :] = 100
    >>> heatmap[:, [0, 1, 2, 3, 11, 12, 13, 14]] = 100
    >>> heatmap = heatmap.fill_diagonal_(0)
    >>> group_experts_on_collaboration_heterogeneous_group(heatmap, num_groups=4, num_fast_groups=2, even_groups=True)
     [[10, 15, 9, 8], [6, 7, 5, 4], [0, 1, 2, 3], [11, 12, 13, 14]]
    """
    num_experts = expert_collab_counts.size(0)
    num_slow_groups = num_groups - num_fast_groups

    if not even_groups:
        raise NotImplementedError("Uneven groups are not supported yet.")
    else:
        group_capacity = num_experts // num_groups

    _check_expert_collab_counts(expert_collab_counts)

    # Create a list of groups
    group_list = []

    # Divide all the experts into two groups: fast and slow
    # 使用熵（entropy）来度量一个专家与多少人都协作过：熵高则协作分布广fast，expert；熵低则协作集中，slow expert；按熵值排序后划分出 fast / slow 的索引列表。
    expert_collab_entropy = compute_entropy(expert_collab_counts)  # (num_experts,)
    sorted_expert_on_entropy = torch.argsort(expert_collab_entropy, descending=False)
    fast_experts = sorted_expert_on_entropy[-num_fast_groups * group_capacity:]
    slow_experts = sorted_expert_on_entropy[:num_slow_groups * group_capacity]

    # 把slow experts先按专家协作对分好
    # 反复从 slow experts 中找协作次数最多的一对 (x, y)；将这对专家作为一对，加入 slow_expert_pairs
    # Group the experts in the slow group
    slow_expert_pairs = []
    slow_expert_stack = slow_experts.clone().tolist()
    while slow_expert_stack:    # 只要还有没配对的专家
        # Find the pair of experts with the highest collaboration counts
        # 找到slow experts的子协作矩阵，找到该子矩阵中协作次数最大的元素位置
        max_idx = torch.argmax(expert_collab_counts[slow_expert_stack][:, slow_expert_stack])
        # 把最大协作次数的位置还原回2D坐标
        x, y = max_idx // len(slow_expert_stack), max_idx % len(slow_expert_stack)
        # 再转回原始专家编号
        x, y = slow_expert_stack[x], slow_expert_stack[y]
        # 把当前找到的协作最多的一对专家 (x, y) 存入结果列表，把x和y从待配对的列表里移除
        slow_expert_pairs.append((x, y))
        slow_expert_stack.remove(x)
        slow_expert_stack.remove(y)

    # Group the slow expert pairs
    # 构建slow expert pair之间的相似度【协作度】矩阵（P,P）-P:专家对数量
    slow_expert_pair_sim_mat = torch.zeros(len(slow_expert_pairs), len(slow_expert_pairs))
    for i, (x, y) in enumerate(slow_expert_pairs):
        for j, (x_, y_) in enumerate(slow_expert_pairs):
            # 内外双层循环遍历所有专家对，专家对之间的协作性/相似度就是两对专家组合的成员之间（4种组合）的协作次数之和的平均值
            # 也就是：如果两对之间的成员互相合作很多，说明它们可以放进同一组。
            slow_expert_pair_sim_mat[i, j] = (expert_collab_counts[x, x_] + expert_collab_counts[y, y_] +
                                              expert_collab_counts[x, y_] + expert_collab_counts[y, x_]) / 4
    # 对矩阵进行对称化处理并把对角线清零
    slow_expert_pair_sim_mat = slow_expert_pair_sim_mat + slow_expert_pair_sim_mat.T    
    slow_expert_pair_sim_mat.fill_diagonal_(0)

    # 调用前面定义的 group_experts_on_collaboration()，对专家对进行分组
    slow_expert_pair_group_list = group_experts_on_collaboration(
        slow_expert_pair_sim_mat, num_slow_groups, even_groups=True)
    
    # 将每组中的 pair 展开还原成专家编号；加入总的 group_list
    for group in slow_expert_pair_group_list:
        group_list.append([slow_expert_pairs[pair_group_id][0] for pair_group_id in group] +
                          [slow_expert_pairs[pair_group_id][1] for pair_group_id in group])

    # sub = expert_collab_counts[fast_experts][:, fast_experts]
    # print("Fast matrix max value:", torch.max(sub))
    # print("Shape:", sub.shape)
    # print("Submatrix:\n", sub)
    # for i in fast_experts:
    #     print(f"Expert {i.item()} collaborations:", expert_collab_counts[i])

    # 对fast_expert，找到fast experts的子协作矩阵，直接那调用前面定义的 group_experts_on_collaboration()，对专家对进行分组
    # Group the experts in the fast group
    fast_expert_group_list = group_experts_on_collaboration(
        expert_collab_counts[fast_experts][:, fast_experts], num_fast_groups, even_groups=True)

    # 将子矩阵索引映射回原始专家编号
    # group 是分组后的结果（每个专家在 fast_experts 里的相对位置）；fast_experts[expert_id] 能取出这个专家在原始矩阵里的编号
    for group in fast_expert_group_list:
        group_list.append([fast_experts[expert_id].item() for expert_id in group])

    return group_list
