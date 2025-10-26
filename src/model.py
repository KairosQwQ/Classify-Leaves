
import torch.nn as nn
import torch.nn.functional as F


#构造的简单CNN  准确率很低（可能是模型复杂度不够，+不是很会调参）
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 32, 112, 112]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 64, 56, 56]
        x = x.view(x.size(0), -1)             # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """基础残差块（用于 ResNet-18/34）"""
    expansion = 1  # 输出通道数扩张倍数（与主分支一致）

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # 第一个卷积层：3x3卷积 + BatchNorm + ReLU
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)  # BatchNorm加速收敛
        # 第二个卷积层：3x3卷积 + BatchNorm（无ReLU，在残差相加后激活）
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # 下采样模块（用于匹配残差分支的维度）

    def forward(self, x):
        identity = x  # 残差分支（捷径连接）

        # 主分支：两次卷积 + 归一化
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 如果需要下采样（ stride>1 或通道数不匹配），对残差分支处理
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接：主分支 + 残差分支
        out += identity
        out = self.relu(out)  # 相加后激活

        return out


class ResNet18(nn.Module):
    """ResNet-18 模型"""
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        # 初始卷积层：7x7大卷积 + 最大池化（减少特征图尺寸）
        self.in_channels = 64  # 初始输入通道数（与第一个残差块匹配）
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 残差层：4个stage，每个stage包含2个基础残差块（共 2x4=8 个残差块）
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)   # 输出通道64
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)  # 输出通道128（下采样）
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)  # 输出通道256（下采样）
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)  # 输出通道512（下采样）

        # 分类头：全局平均池化 + 全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 无论输入尺寸，输出(1,1)
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """构建残差层（包含多个残差块）"""
        downsample = None
        # 如果 stride>1 或输入输出通道数不匹配，需要下采样调整残差分支
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, out_channels * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        # 第一个残差块：可能包含下采样
        layers.append(block(
            self.in_channels, out_channels, stride, downsample
        ))
        self.in_channels = out_channels * block.expansion  # 更新输入通道数

        # 后续残差块：无需下采样（stride=1）
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始特征提取
        x = self.conv1(x)       # [B, 3, 224, 224] → [B, 64, 112, 112]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # [B, 64, 112, 112] → [B, 64, 56, 56]

        # 残差层特征提取
        x = self.layer1(x)      # [B, 64, 56, 56] → [B, 64, 56, 56]（无下采样）
        x = self.layer2(x)      # [B, 64, 56, 56] → [B, 128, 28, 28]（下采样）
        x = self.layer3(x)      # [B, 128, 28, 28] → [B, 256, 14, 14]（下采样）
        x = self.layer4(x)      # [B, 256, 14, 14] → [B, 512, 7, 7]（下采样）

        # 分类头
        x = self.avgpool(x)     # [B, 512, 7, 7] → [B, 512, 1, 1]
        x = torch.flatten(x, 1) # [B, 512, 1, 1] → [B, 512]
        x = self.fc(x)          # [B, 512] → [B, num_classes]

        return x