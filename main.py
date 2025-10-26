import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

from src.data import LeaveDataset
from src.model import SimpleCNN,ResNet18 
from src.train import train_model

# -----------------------------
# 数据加载与预处理
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

""" 替换为你的路径 """
train_file = r"E:\DataSet\classify-leaves\train.csv" 
img_dir = r"E:\DataSet\classify-leaves"

full_dataset = LeaveDataset(csv_file=train_file, dir=img_dir, transform=train_transform)

# 分层划分数据集（确保每个类别都出现在训练与验证中）
labels = full_dataset.data['label'].map(full_dataset.label2idx).values
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, val_idx in splitter.split(np.zeros(len(labels)), labels):
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

# -----------------------------
# 超参数
batch_size = 64
num_epochs = 50
learning_rate = 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = "./best_model.pth"

#构造dataloader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"device:{device}, Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
# -----------------------------
# 模型、损失、优化器
num_classes = len(full_dataset.label2idx)
#把模型迁移到device上计算
model = ResNet18(num_classes=num_classes).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# -----------------------------
# 启动训练
train_model(model, train_loader, val_loader, criterion, optimizer, device,save_path, num_epochs)
