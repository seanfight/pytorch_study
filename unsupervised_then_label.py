"""
简单示例：先无监督学习，再添加标签做分类

流程：
1) 无监督：用自编码器学习数据表示（不需要标签）
2) 添加标签：冻结编码器，接上分类头，用少量标签做监督微调
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


torch.manual_seed(42)
device = torch.device("cpu")


# ---------- 1. 构造模拟数据（3 类二维点云） ----------
def make_blobs(n_per_class: int = 200):
    centers = torch.tensor([[0.0, 0.0], [3.0, 3.0], [-3.0, 3.0]])
    xs, ys = [], []
    for label, center in enumerate(centers):
        points = center + 0.6 * torch.randn(n_per_class, 2)
        xs.append(points)
        ys.append(torch.full((n_per_class,), label, dtype=torch.long))
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    perm = torch.randperm(x.size(0))
    return x[perm], y[perm]


x_all, y_all = make_blobs()
n_train = int(0.8 * len(x_all))
x_train, y_train = x_all[:n_train], y_all[:n_train]
x_test, y_test = x_all[n_train:], y_all[n_train:]

# 无监督阶段：只用特征，不用标签
unsup_loader = DataLoader(TensorDataset(x_train), batch_size=64, shuffle=True)

# 监督阶段：只用一小部分带标签样本（模拟标注成本高）
n_labeled = 60
labeled_loader = DataLoader(
    TensorDataset(x_train[:n_labeled], y_train[:n_labeled]),
    batch_size=16,
    shuffle=True,
)


# ---------- 2. 自编码器（无监督） ----------
class AutoEncoder(nn.Module):
    def __init__(self, in_dim=2, hidden=16, latent=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


ae = AutoEncoder().to(device)
ae_opt = torch.optim.Adam(ae.parameters(), lr=1e-2)
recon_loss_fn = nn.MSELoss()

print("=== 阶段 1：无监督预训练（重建损失）===")
ae.train()
for epoch in range(1, 41):
    total = 0.0
    for (batch_x,) in unsup_loader:
        batch_x = batch_x.to(device)
        recon, _ = ae(batch_x)
        loss = recon_loss_fn(recon, batch_x)
        ae_opt.zero_grad()
        loss.backward()
        ae_opt.step()
        total += loss.item() * batch_x.size(0)
    if epoch % 10 == 0:
        print(f"epoch {epoch:02d}  recon_loss={total / len(x_train):.4f}")


# ---------- 3. 添加标签：在编码器上接分类头 ----------
class LabeledModel(nn.Module):
    def __init__(self, encoder: nn.Module, latent=4, n_classes=3):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(latent, n_classes)

    def forward(self, x):
        z = self.encoder(x)
        return self.classifier(z)


# 冻结编码器，只训练新加的分类层（体现“先无监督学表示，再加标签”）
for p in ae.encoder.parameters():
    p.requires_grad = False

model = LabeledModel(ae.encoder).to(device)
clf_opt = torch.optim.Adam(model.classifier.parameters(), lr=1e-2)
ce_loss_fn = nn.CrossEntropyLoss()


@torch.no_grad()
def accuracy(net, x, y):
    net.eval()
    pred = net(x.to(device)).argmax(dim=1).cpu()
    return (pred == y).float().mean().item()


print("\n=== 阶段 2：添加标签后微调分类头 ===")
print(f"仅使用 {n_labeled} 条带标签样本")
model.train()
for epoch in range(1, 51):
    total = 0.0
    for batch_x, batch_y in labeled_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        logits = model(batch_x)
        loss = ce_loss_fn(logits, batch_y)
        clf_opt.zero_grad()
        loss.backward()
        clf_opt.step()
        total += loss.item() * batch_x.size(0)
    if epoch % 10 == 0:
        train_acc = accuracy(model, x_train[:n_labeled], y_train[:n_labeled])
        test_acc = accuracy(model, x_test, y_test)
        print(
            f"epoch {epoch:02d}  ce_loss={total / n_labeled:.4f}  "
            f"labeled_acc={train_acc:.3f}  test_acc={test_acc:.3f}"
        )

print("\n完成：无监督表示学习 -> 加分类头用标签预测类别")
