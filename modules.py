import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiInterestEncoder(nn.Module):
    def __init__(self, emb_dim, num_interests=4):
        """
        多兴趣建模模块：将一个用户的行为序列编码为 K 个兴趣向量

        参数:
            emb_dim: 物品嵌入维度
            num_interests: 要提取的兴趣数量 K
        """
        super(MultiInterestEncoder, self).__init__()
        self.num_interests = num_interests
        self.query = nn.Parameter(torch.randn(num_interests, emb_dim))  # [K, D]
        self.attn_layer = nn.Linear(emb_dim, num_interests)

    def forward(self, item_seq_emb):
        """
        前向传播

        输入:
            item_seq_emb: Tensor，shape = [B, L, D]，用户历史物品嵌入序列
        输出:
            multi_interest: Tensor，shape = [B, K, D]，每个用户的 K 个兴趣表示
        """
        # item_seq_emb: [B, L, D]
        attn_scores = self.attn_layer(item_seq_emb)  # [B, L, K]
        attn_weights = F.softmax(attn_scores, dim=1)  # 对 L 做 softmax，shape = [B, L, K]

        # 对历史序列加权平均：sum_{l}(w_lk * x_l) → 对每个兴趣方向
        multi_interest = torch.einsum('blk,bld->bkd', attn_weights, item_seq_emb)  # [B, K, D]

        return multi_interest
