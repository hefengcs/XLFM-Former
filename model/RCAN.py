import torch
from torch import nn

class RCAN_Net(nn.Module):
    def __init__(self, input_channels=28, output_channels=300, num_features=64, num_rg=5, num_rcab=10, reduction=16):
        """
        使用 RCAN 实现的 RCAN_Net 模型，适配光场数据重建。

        Args:
            input_channels (int): 输入通道数（默认为 28）。
            output_channels (int): 输出通道数（默认为 300）。
            num_features (int): RCAN 的特征数。
            num_rg (int): 残差组（RG）的数量。
            num_rcab (int): 每个 RG 中的 RCAB 数量。
            reduction (int): 通道注意力的缩减比。
        """
        super(RCAN_Net, self).__init__()

        self.model = RCAN(input_channels, output_channels, num_features, num_rg, num_rcab, reduction)

    def forward(self, x):
        x = self.model(x)
        return x


class RCAN(nn.Module):
    def __init__(self, input_channels, output_channels, num_features, num_rg, num_rcab, reduction):
        super(RCAN, self).__init__()
        # 输入层：将输入映射到特征空间
        self.sf = nn.Conv2d(input_channels, num_features, kernel_size=3, padding=1)

        # 残差组：包含多个 RG 模块
        self.rgs = nn.Sequential(*[RG(num_features, num_rcab, reduction) for _ in range(num_rg)])

        # 输出层：将特征映射到目标通道
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_features, output_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.sf(x)  # 初始特征提取
        residual = x    # 残差
        x = self.rgs(x) # 残差组处理
        x = self.conv1(x)
        x += residual   # 残差连接
        x = self.conv2(x)  # 输出特征
        return x


class RG(nn.Module):
    def __init__(self, num_features, num_rcab, reduction):
        super(RG, self).__init__()
        # 一个残差组中包含多个 RCAB 模块
        self.module = [RCAB(num_features, reduction) for _ in range(num_rcab)]
        self.module.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return x + self.module(x)


class RCAB(nn.Module):
    def __init__(self, num_features, reduction):
        super(RCAB, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            ChannelAttention(num_features, reduction)
        )

    def forward(self, x):
        return x + self.module(x)


class ChannelAttention(nn.Module):
    def __init__(self, num_features, reduction):
        super(ChannelAttention, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.module(x)
def main():
    # 初始化模型
    input_channels = 28
    output_channels = 300
    model = RCAN_Net(input_channels=28, output_channels=300, num_features=64, num_rg=5, num_rcab=10, reduction=16)

    print("Model structure:")
    print(model)

    # 测试输入和输出
    input_tensor = torch.randn(1, 28, 600, 600)  # 输入形状 [batch_size, channels, height, width]
    output = model(input_tensor)

    # 打印输入和输出的形状
    print("Input shape:", input_tensor.shape)   # [1, 28, 600, 600]
    print("Output shape:", output.shape)       # [1, 300, 600, 600]


if __name__ == "__main__":
    main()
