import torch
import torch.nn as nn

# 定义门控机制类
class GateMechanism(nn.Module):
    def __init__(self, input_dim):
        super(GateMechanism, self).__init__()
        self.gate_layer = nn.Linear(input_dim * 2, 1)  # 输入维度是两个特征拼接后的维度，输出维度是1

    def forward(self, input_A, input_B):
        # 将A和B的特征拼接起来
        combined_input = torch.cat((input_A, input_B), dim=-1)
        # 计算门控单元的输出
        gate_output = torch.sigmoid(self.gate_layer(combined_input))
        # 应用门控单元的输出，融合A和B的特征
        fused_feature = gate_output * input_A + (1 - gate_output) * input_B
        return fused_feature

# 示例输入
A = torch.randn(4, 128, 768)
B = torch.randn(4, 128, 768)

# 创建门控机制对象
gate_mechanism = GateMechanism(input_dim=768)

# 使用门控机制融合A和B的特征
fused_feature = gate_mechanism(A, B)

# 查看融合后特征的形状
print(fused_feature.shape)  # 输出：torch.Size([4, 128, 768])
