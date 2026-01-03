import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        return x


class ScaleDot(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return x * self.scale


def haar_dwt(x):
    B, C, H, W = x.shape
    if H % 2 != 0 or W % 2 != 0:
        x = F.pad(x, (0, W % 2, 0, H % 2))

    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]

    LL = (x00 + x01 + x10 + x11) / 2
    LH = (x00 - x01 + x10 - x11) / 2
    HL = (x00 + x01 - x10 - x11) / 2
    HH = (x00 - x01 - x10 + x11) / 2

    return torch.cat([LL, LH, HL, HH], dim=1)  # (B, 4C, H/2, W/2)


def haar_idwt(x_dwt):
    B, C4, H_half, W_half = x_dwt.shape
    C = C4 // 4
    LL, LH, HL, HH = torch.split(x_dwt, C, dim=1)

    x00 = (LL + LH + HL + HH) / 2
    x01 = (LL - LH + HL - HH) / 2
    x10 = (LL + LH - HL - HH) / 2
    x11 = (LL - LH - HL + HH) / 2

    out = torch.zeros(B, C, H_half * 2, W_half * 2, device=x_dwt.device)
    out[:, :, 0::2, 0::2] = x00
    out[:, :, 0::2, 1::2] = x01
    out[:, :, 1::2, 0::2] = x10
    out[:, :, 1::2, 1::2] = x11

    return out


class GMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),  # [新增] 20% 的神经元随机失活
            nn.Linear(hidden_dim, dim),
            nn.Dropout(0.2)   # [新增]
        )

    def forward(self, x):
        return self.net(x)


# 将此代码块放入 model_rswa.py 中，替换原有的 RSWABlock 类

class RSWABlock(nn.Module):
    def __init__(self, dim, window_size, num_heads=4, recon_weight=0.1):
        super().__init__()
        self.dim = dim
        self.ws = window_size
        self.recon_weight = recon_weight

        self.preprocess = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        )
        self.norm = LayerNorm2d(dim)

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        self.gmlp = GMLP(dim, dim * 4)

        # 重建头：尝试将融合后的特征还原回原始的 patch 特征
        self.recon_head = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x

        x_feat = self.preprocess(x)
        x_feat = self.norm(x_feat)

        # Padding logic to fit windows
        pad_h = (self.ws - H % self.ws) % self.ws
        pad_w = (self.ws - W % self.ws) % self.ws
        x_feat = F.pad(x_feat, (0, pad_w, 0, pad_h))

        # [Sliding Window 核心] 使用 unfold 提取滑动窗口
        # patches shape: (B, C, nH, nW, ws, ws)
        patches = x_feat.unfold(2, self.ws, self.ws).unfold(3, self.ws, self.ws)
        nH, nW = patches.shape[2], patches.shape[3]

        # 调整 shape 准备进 Attention: (B * nH * nW, ws*ws, C)
        patches = patches.contiguous().view(B, C, nH, nW, self.ws, self.ws)
        patches = patches.permute(0, 2, 3, 4, 5, 1).contiguous().view(-1, self.ws * self.ws, C)

        # Attention 机制
        qkv = self.qkv(patches)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(patches.shape[0], patches.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(patches.shape[0], patches.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(patches.shape[0], patches.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out_attn = (attn @ v).transpose(1, 2).reshape(patches.shape[0], patches.shape[1], C)

        # gMLP 分支
        v_flat = v.transpose(1, 2).reshape(patches.shape[0], patches.shape[1], C)
        out_gmlp = self.gmlp(v_flat)

        # 融合
        out_fused = out_attn * out_gmlp
        out_fused = self.proj(out_fused)

        recon_pred = self.recon_head(out_fused)

        # 使用 L1 Loss 计算差异
        recon_loss = F.l1_loss(recon_pred, patches.detach()) * self.recon_weight

        # 还原回图片尺寸
        out_fused = out_fused.view(B, nH, nW, self.ws, self.ws, C)
        out_fused = out_fused.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, nH * self.ws, nW * self.ws)

        if pad_h > 0 or pad_w > 0:
            out_fused = out_fused[:, :, :H, :W]

        # 返回特征和辅助损失
        return out_fused + shortcut, {'recon_loss': recon_loss}

class ResNetClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        def conv3x3(in_planes, out_planes, stride=1):
            return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

        class BasicBlock(nn.Module):
            expansion = 1

            def __init__(self, inplanes, planes, stride=1, downsample=None):
                super().__init__()
                self.conv1 = conv3x3(inplanes, planes, stride)
                self.bn1 = nn.BatchNorm2d(planes)
                self.relu = nn.ReLU(inplace=True)
                self.conv2 = conv3x3(planes, planes)
                self.bn2 = nn.BatchNorm2d(planes)
                self.downsample = downsample
                self.stride = stride

            def forward(self, x):
                identity = x
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                out = self.conv2(out)
                out = self.bn2(out)
                if self.downsample is not None:
                    identity = self.downsample(x)
                out += identity
                out = self.relu(out)
                return out

        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class AIGCDetector(nn.Module):
    def __init__(self, num_classes=2, embed_dim=96):
        super().__init__()
        self.embed_dim = embed_dim

        self.dwt_proj = nn.Conv2d(12, embed_dim, 1)
        self.dwt_block = RSWABlock(embed_dim, window_size=4)
        self.idwt_proj = nn.Conv2d(embed_dim, 12, 1)

        self.fft_proj = nn.Conv2d(3, embed_dim, 1)
        self.fft_block = RSWABlock(embed_dim, window_size=8)
        self.ifft_proj = nn.Conv2d(embed_dim, 3, 1)
        self.scale_dot = ScaleDot(3)

        self.lambda_fuse = 0.4

        self.classifier = ResNetClassifier(in_channels=3, num_classes=num_classes)

    def forward(self, x):
        x_dwt = haar_dwt(x)
        feat_dwt = self.dwt_proj(x_dwt)
        feat_dwt, aux_dwt = self.dwt_block(feat_dwt)

        feat_dwt_back = self.idwt_proj(feat_dwt)
        img_recon_dwt = haar_idwt(feat_dwt_back)
        img_recon_dwt = F.interpolate(img_recon_dwt, size=x.shape[-2:], mode='bilinear', align_corners=False)

        x_float32 = x.float()
        fft_x = torch.fft.fft2(x_float32)
        amp = torch.abs(fft_x)
        phase = torch.angle(fft_x)

        feat_fft = self.fft_proj(phase)
        feat_fft, aux_fft = self.fft_block(feat_fft)

        new_phase = self.ifft_proj(feat_fft)
        new_phase_float32 = new_phase.float()
        complex_spec = torch.polar(amp, new_phase_float32)
        img_recon_fft = torch.fft.ifft2(complex_spec).real
        img_recon_fft = img_recon_fft.to(x.dtype)
        img_recon_fft = self.scale_dot(img_recon_fft)

        X_fused = (1 - self.lambda_fuse) * img_recon_dwt + self.lambda_fuse * img_recon_fft

        logits = self.classifier(X_fused)

        return logits, aux_dwt['recon_loss'] + aux_fft['recon_loss']

