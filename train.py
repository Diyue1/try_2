import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import average_precision_score, accuracy_score
from model_rswa import AIGCDetector

# 路径与超参数对齐论文 [cite: 221, 245, 246]
TRAIN_DIR = "./newTrain/train" # 包含 4-class 设置: car, cat, chair, horse
VAL_DIR = "./newTrain/val"
BATCH_SIZE = 128
LR = 2e-4
EPOCHS = 90
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BinaryDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        # 论文特定的 4-class 训练设置 [cite: 221]
        target_cats = ['car', 'cat', 'chair', 'horse']
        for root, _, files in os.walk(root_dir):
            if any(cat in root for cat in target_cats):
                if '0_real' in root or '1_fake' in root:
                    label = 0 if '0_real' in root else 1
                    for f in files:
                        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.samples.append((os.path.join(root, f), label))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, label

def train():
    # 数据增强对齐论文逻辑 [cite: 193]
    train_tf = transforms.Compose([
        transforms.RandomCrop((256, 256), pad_if_needed=True),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_tf = transforms.Compose([
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(BinaryDataset(TRAIN_DIR, train_tf), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(BinaryDataset(VAL_DIR, val_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = AIGCDetector().to(DEVICE)
    # Adam 优化器 [cite: 245]
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # 学习率每 10 轮衰减 20% [cite: 247]
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    # 损失函数对齐 BCELoss 
    criterion = nn.BCELoss()

    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).float()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

        scheduler.step()

        # 验证
        model.eval()
        y_true, y_prob = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs.to(DEVICE))
                y_true.extend(labels.numpy())
                y_prob.extend(outputs.cpu().numpy())

        y_pred = [1 if p > 0.5 else 0 for p in y_prob]
        acc = accuracy_score(y_true, y_pred) * 100
        ap = average_precision_score(y_true, y_prob) * 100
        print(f"Epoch {epoch+1} | Acc: {acc:.2f}% | AP: {ap:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_model_paper.pth")

if __name__ == "__main__":
    train()

