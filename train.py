import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageFile
from tqdm import tqdm
from sklearn.metrics import average_precision_score, accuracy_score
import numpy as np
import random

from model_rswa import AIGCDetector

ImageFile.LOAD_TRUNCATED_IMAGES = True

# 路径请保持不用变
TRAIN_DIR = "/data/ziqiang/yjz/dataset/Benchmark/newTrain/train"
VAL_DIR = "/data/ziqiang/yjz/dataset/Benchmark/newTrain/val"

PHYSICAL_BATCH_SIZE = 64
TARGET_BATCH_SIZE = 128
ACCUM_STEPS = TARGET_BATCH_SIZE // PHYSICAL_BATCH_SIZE

LR = 2e-4
EPOCHS = 20
NUM_WORKERS = 4  # 建议改为 4 提高加载速度，如果报错改回 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# [新增] 自定义 JPEG 压缩增强（模拟网络传播图）
class RandomJPEGCompression:
    def __init__(self, quality_min=60, quality_max=100, p=0.5):
        self.quality_min = quality_min
        self.quality_max = quality_max
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            quality = random.randint(self.quality_min, self.quality_max)
            img = img.convert('RGB')
            # 在内存中压缩再读取
            import io
            buffer = io.BytesIO()
            img.save(buffer, "JPEG", quality=quality)
            return Image.open(buffer)
        return img


# [修改] 增强的数据加载器
class RecursiveBinaryDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {'0_real': 0, '1_fake': 1}

        if not os.path.exists(root_dir):
            raise RuntimeError(f"路径不存在: {root_dir}")

        for root, dirs, files in os.walk(root_dir):
            folder_name = os.path.basename(root)
            if folder_name in self.class_to_idx:
                label = self.class_to_idx[folder_name]
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.webp')):
                        path = os.path.join(root, file)
                        self.samples.append((path, label))

        # 随机打乱数据，防止按文件夹顺序读取
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            # 遇到坏图返回全黑张量，不报错中断
            fallback = torch.zeros((3, 256, 256))
            return fallback, label


def train_model():
    print(f"环境: {torch.cuda.get_device_name(0)}")

    # [重点修改] 强力数据增强：防止过拟合的核心
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        RandomJPEGCompression(quality_min=50, quality_max=95, p=0.5),  # 模拟压缩
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.RandomRotation(degrees=15),  # 随机旋转
        transforms.ColorJitter(brightness=0.1, contrast=0.1),  # 颜色微扰
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # 随机模糊
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 验证集保持干净，只做 Resize 和 Normalize
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = RecursiveBinaryDataset(TRAIN_DIR, transform=train_transform)
    val_ds = RecursiveBinaryDataset(VAL_DIR, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=PHYSICAL_BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=PHYSICAL_BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    model = AIGCDetector(num_classes=2, embed_dim=96).to(DEVICE)

    # [修改] 加入 weight_decay (L2正则化)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # 调整学习率策略
    criterion = nn.CrossEntropyLoss()

    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    best_acc = 0.0
    print(f"开始训练... (增强版)")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for i, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs, recon_loss = model(inputs)
                    cls_loss = criterion(outputs, labels)
                    loss = (cls_loss + recon_loss) / ACCUM_STEPS
            else:
                outputs, recon_loss = model(inputs)
                cls_loss = criterion(outputs, labels)
                loss = (cls_loss + recon_loss) / ACCUM_STEPS

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % ACCUM_STEPS == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * ACCUM_STEPS
            if i % 50 == 0:
                pbar.set_postfix(loss=loss.item() * ACCUM_STEPS)

        scheduler.step()

        # 验证部分
        model.eval()
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, _ = model(inputs)
                else:
                    outputs, _ = model(inputs)

                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_targets.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().float().numpy())

        all_preds = [1 if p > 0.5 else 0 for p in all_probs]
        val_acc = accuracy_score(all_targets, all_preds) * 100
        val_ap = average_precision_score(all_targets, all_probs) * 100

        print(
            f"Epoch {epoch + 1} | Loss: {running_loss / len(train_loader):.4f} | Acc: {val_acc:.2f}% | AP: {val_ap:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_rswa_model_v2.pth")
            print("模型已保存")


if __name__ == "__main__":
    train_model()