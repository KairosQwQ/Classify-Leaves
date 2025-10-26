import torch
from tqdm import tqdm #能可视化训练进程
import matplotlib.pyplot as plt


# -----------------------------
# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, device, save_path, num_epochs=10):
    # 记录准确率的列表
    train_losses = []
    val_losses = []
    train_accs = []  # 训练集准确率
    val_accs = []    # 验证集准确率
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # -----------------------------
        # 训练阶段
        model.train() #设置模型为训练模式
        running_loss = 0.0
        correct = 0  
        total = 0   

        # 构造python迭代器  方便循环访问 （tqdm就这么用好了）
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        
        
        # 训练循环
        for images, labels in train_bar:
            
            images, labels = images.to(device), labels.to(device)#将输入X与label y迁移到 指定device
            optimizer.zero_grad()#梯度清零 避免累加

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)  # 取预测概率最大的类别索引
            total += labels.size(0)                    # 累加总样本数
            correct += (predicted == labels).sum().item()  # 累加正确样本数

            # 反向传播与优化
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            # 在进度条显示当前批次的损失和准确率
            train_bar.set_postfix(loss=loss.item(), acc=correct/total)

        # 计算当前 epoch 的训练集平均损失和准确率
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct / total  
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc) 

        # -----------------------------
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0 
        val_total = 0   

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                # 计算验证集准确率
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                val_loss += loss.item() * images.size(0)
                # 显示当前批次的损失和准确率
                val_bar.set_postfix(loss=loss.item(), acc=val_correct/val_total)

        # 计算当前 epoch 的验证集平均损失和准确率
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)  

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")

        # 保存最优模型（以验证集损失为指标）
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model at epoch {epoch+1} (Val Acc: {epoch_val_acc:.4f})")

    # -----------------------------
    # 绘制损失曲线和准确率曲线
    plt.figure(figsize=(12, 5))

    # 子图1：损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()

    # 子图2：准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accs, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()

    plt.tight_layout()  # 调整子图间距
    plt.show()

    return train_losses, val_losses, train_accs, val_accs