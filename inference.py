import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

from tqdm import tqdm
from src.data import LeaveDataset
from src.model import ResNet18

def load_model(model_path, num_classes, device):
    """加载训练好的模型"""
    model = ResNet18(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def get_label_mapping(train_csv_path):
    """从训练集CSV文件中获取标签映射"""
    train_data = pd.read_csv(train_csv_path)
    classes = sorted(train_data['label'].unique())
    label2idx = {cls_name: i for i, cls_name in enumerate(classes)}
    idx2label = {i: cls_name for cls_name, i in label2idx.items()}
    return label2idx, idx2label

def inference_on_test_set(test_csv_path, img_dir, model_path, train_csv_path, 
                         batch_size=64, device=None):
    """
    对测试集进行推理并生成submission.csv
    
    Args:
        test_csv_path: 测试集CSV文件路径
        img_dir: 图片目录路径
        model_path: 训练好的模型权重路径
        train_csv_path: 训练集CSV文件路径（用于获取标签映射）
        batch_size: 批次大小
        device: 设备（CPU/GPU）
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
 
    # 获取标签映射
    print("加载标签映射...")
    label2idx, idx2label = get_label_mapping(train_csv_path)
    num_classes = len(label2idx)
    print(f"类别数量: {num_classes}")
    
    # 定义测试时的数据变换（不包含数据增强）
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 加载测试数据集
    print("加载测试数据集...")
    test_dataset = LeaveDataset(csv_file=test_csv_path, dir=img_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 加载模型
    print("加载训练好的模型...")
    model = load_model(model_path, num_classes, device)
    
    print("开始推理...")
    predictions = []
    image_names = []
    
    with torch.no_grad():
        for images, names in tqdm(test_loader, desc="推理进度"):
            images = images.to(device)
            
            # 前向传播
            outputs = model(images)
            # 获取预测结果（概率最大的类别）
            _, predicted = torch.max(outputs, 1)     
            # 将预测结果转换为标签名称
            for pred_idx in predicted.cpu().numpy():
                pred_label = idx2label[pred_idx]
                predictions.append(pred_label)      
            # 保存图片名称
            image_names.extend(names)
    
    # 创建submission DataFrame
    submission_df = pd.DataFrame({
        'image': image_names,
        'label': predictions
    })
    
    # 保存submission文件
    submission_path = 'submission.csv'
    submission_df.to_csv(submission_path, index=False)    
    
    return submission_df

if __name__ == "__main__":
    # 配置路径
    test_csv_path = r"E:\DataSet\classify-leaves\test.csv"  # 测试集CSV文件路径
    img_dir = r"E:\DataSet\classify-leaves"  # 图片目录路径
    model_path = "./best_model.pth"  # 训练好的模型权重路径
    train_csv_path = r"E:\DataSet\classify-leaves\train.csv"  # 训练集CSV文件路径（用于获取标签映射） 
    # 执行推理

    submission_df = inference_on_test_set(
        test_csv_path=test_csv_path,
        img_dir=img_dir,
        model_path=model_path,
        train_csv_path=train_csv_path,
        batch_size=64
    )
   
