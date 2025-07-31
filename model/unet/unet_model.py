""" Full assembly of the parts to form the complete network """

from model.unet.unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 编码器部分
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # 解码器部分
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # 输出层，输出通道数设置为 n_classes（300）
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # 编码阶段
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 解码阶段
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # 输出层
        logits = self.outc(x)
        return logits


if __name__ == "__main__":
    # 输入和输出配置
    n_channels = 28  # 输入通道数
    n_classes = 300  # 输出通道数
    model = UNet(n_channels, n_classes, bilinear=True)

    # 测试输入数据
    input_tensor = torch.randn(1, 28, 600, 600)  # 输入张量，形状为 [batch_size, channels, height, width]
    output = model(input_tensor)

    # 打印模型结构和输出形状
    print(model)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output.shape)
