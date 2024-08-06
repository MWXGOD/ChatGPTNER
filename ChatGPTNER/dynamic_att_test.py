import torch
import torch.nn as nn

# 自定义注意力机制
class DynamicAttention(nn.Module):
    def __init__(self, num_sources):
        super(DynamicAttention, self).__init__()
        self.num_sources = num_sources
        self.attention_weights = nn.Parameter(torch.ones(num_sources + 1))  # +1 for BERT

    def forward(self, source_features, bert_output):
        # 计算注意力分数
        attention_scores = []
        for i in range(self.num_sources):
            score = torch.sum(self.attention_weights[i] * source_features[i], dim=1)
            attention_scores.append(score)

        # 加入BERT的输出
        bert_score = torch.sum(self.attention_weights[-1] * bert_output, dim=1)
        attention_scores.append(bert_score)

        # 应用softmax
        attention_weights_normalized = torch.softmax(torch.stack(attention_scores, dim=1), dim=1)

        # 加权融合
        weighted_sum = torch.sum(torch.stack(
            [attention_weights_normalized[:, i:i + 1] * source_features[i] for i in range(self.num_sources)], dim=2),
                                 dim=2)

        return weighted_sum


# 示例三个尺度的特征
num_sources = 3
source_features = [torch.randn(32, 768), torch.randn(32, 768), torch.randn(32, 768)]  # 每个尺度特征为768维

# 示例BERT的输出
bert_output = torch.randn(32, 768)  # 假设BERT的输出也是768维

# 创建动态注意力模型
dynamic_attention = DynamicAttention(num_sources)

# 融合多尺度特征和BERT的输出
merged_feature = dynamic_attention(source_features, bert_output)

# 输出融合后的特征
print(merged_feature)
