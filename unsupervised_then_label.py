# 简单示例：先无监督学习，再添加标签做分类
# 流程 1）无监督：用自编码器学习数据表示（不需要标签）
# 流程 2）添加标签：冻结编码器，接上分类头，用少量标签做监督微调
# ---------------------------------------------------------------------------
import torch  # 导入 PyTorch 主库
import torch.nn as nn  # 导入神经网络模块，方便定义层和模型
from torch.utils.data import DataLoader, TensorDataset  # 导入数据批处理和张量数据集工具
# ---------------------------------------------------------------------------
torch.manual_seed(42)  # 固定随机种子，保证每次运行结果可复现
device = torch.device("cpu")  # 指定计算设备为 CPU（简单示例不需要 GPU）
# ---------------------------------------------------------------------------
# ---------- 1. 构造模拟数据（3 类二维点云） ----------
def make_blobs(n_per_class: int = 200):  # 定义生成模拟点云数据的函数，默认每类 200 个点
    centers = torch.tensor([[0.0, 0.0], [3.0, 3.0], [-3.0, 3.0]])  # 三类数据的中心坐标
    xs, ys = [], []  # 分别存放所有特征和对应标签的列表
    for label, center in enumerate(centers):  # 遍历每个类别编号及其中心点
        points = center + 0.6 * torch.randn(n_per_class, 2)  # 在中心附近加噪声生成二维样本点
        xs.append(points)  # 把当前类的特征加入特征列表
        ys.append(torch.full((n_per_class,), label, dtype=torch.long))  # 为当前类生成相同标签向量
    x = torch.cat(xs, dim=0)  # 把三类特征在第 0 维拼接成完整特征矩阵
    y = torch.cat(ys, dim=0)  # 把三类标签拼接成完整标签向量
    perm = torch.randperm(x.size(0))  # 生成打乱样本顺序的随机索引
    return x[perm], y[perm]  # 按随机索引返回打乱后的特征和标签
# ---------------------------------------------------------------------------
x_all, y_all = make_blobs()  # 生成全部模拟数据
n_train = int(0.8 * len(x_all))  # 按 8:2 划分，计算训练集样本数
x_train, y_train = x_all[:n_train], y_all[:n_train]  # 取出训练集特征和标签
x_test, y_test = x_all[n_train:], y_all[n_train:]  # 取出测试集特征和标签
# 无监督阶段：只用特征，不用标签
unsup_loader = DataLoader(TensorDataset(x_train), batch_size=64, shuffle=True)  # 无监督数据加载器，按批打乱取特征
# 监督阶段：只用一小部分带标签样本（模拟标注成本高）
n_labeled = 60  # 仅使用 60 条带标签样本进行监督微调
labeled_loader = DataLoader(  # 创建带标签数据的加载器
    TensorDataset(x_train[:n_labeled], y_train[:n_labeled]),  # 取前 n_labeled 条特征和标签组成数据集
    batch_size=16,  # 监督阶段每批取 16 条样本
    shuffle=True,  # 每个 epoch 打乱监督样本顺序
)  # 带标签 DataLoader 定义结束
# ---------------------------------------------------------------------------
# ---------- 2. 自编码器（无监督） ----------
class AutoEncoder(nn.Module):  # 定义自编码器模型类，继承自 nn.Module
    def __init__(self, in_dim=2, hidden=16, latent=4):  # 初始化：输入维、隐藏维、潜在表示维
        super().__init__()  # 调用父类构造函数，完成 Module 初始化
        self.encoder = nn.Sequential(  # 定义编码器：把输入压缩成低维表示
            nn.Linear(in_dim, hidden),  # 全连接层：输入维 -> 隐藏维
            nn.ReLU(),  # ReLU 非线性激活，增强表达能力
            nn.Linear(hidden, latent),  # 全连接层：隐藏维 -> 潜在表示维
        )  # 编码器 Sequential 结束
        self.decoder = nn.Sequential(  # 定义解码器：把潜在表示还原回输入空间
            nn.Linear(latent, hidden),  # 全连接层：潜在表示维 -> 隐藏维
            nn.ReLU(),  # ReLU 非线性激活
            nn.Linear(hidden, in_dim),  # 全连接层：隐藏维 -> 原始输入维
        )  # 解码器 Sequential 结束
    def forward(self, x):  # 定义前向传播
        z = self.encoder(x)  # 编码得到潜在表示 z
        return self.decoder(z), z  # 返回重建结果和解码前的潜在表示
# ---------------------------------------------------------------------------
ae = AutoEncoder().to(device)  # 创建自编码器并放到指定设备
ae_opt = torch.optim.Adam(ae.parameters(), lr=1e-2)  # 用 Adam 优化自编码器全部参数
recon_loss_fn = nn.MSELoss()  # 用均方误差衡量重建质量
print("=== 阶段 1：无监督预训练（重建损失）===")  # 打印阶段 1 标题
ae.train()  # 将自编码器切换到训练模式
for epoch in range(1, 41):  # 无监督训练 40 个 epoch
    total = 0.0  # 累计当前 epoch 的总重建损失
    for (batch_x,) in unsup_loader:  # 按批取出无标签特征
        batch_x = batch_x.to(device)  # 把当前批次特征放到设备上
        recon, _ = ae(batch_x)  # 前向传播得到重建结果（忽略潜在表示）
        loss = recon_loss_fn(recon, batch_x)  # 计算重建值与原始输入的 MSE 损失
        ae_opt.zero_grad()  # 清空上一轮累积的梯度
        loss.backward()  # 反向传播，计算参数梯度
        ae_opt.step()  # 根据梯度更新自编码器参数
        total += loss.item() * batch_x.size(0)  # 按样本数加权累计损失，便于算平均
    if epoch % 10 == 0:  # 每 10 个 epoch 打印一次进度
        print(f"epoch {epoch:02d}  recon_loss={total / len(x_train):.4f}")  # 打印平均重建损失
# ---------------------------------------------------------------------------
# ---------- 3. 添加标签：在编码器上接分类头 ----------
class LabeledModel(nn.Module):  # 定义带标签分类模型：编码器 + 分类头
    def __init__(self, encoder: nn.Module, latent=4, n_classes=3):  # 传入已训练编码器、潜在维、类别数
        super().__init__()  # 调用父类构造函数
        self.encoder = encoder  # 复用无监督阶段学到的编码器
        self.classifier = nn.Linear(latent, n_classes)  # 新建线性分类头：潜在表示 -> 类别 logits
    def forward(self, x):  # 定义分类模型的前向传播
        z = self.encoder(x)  # 先用编码器提取特征表示
        return self.classifier(z)  # 再用分类头输出各类别分数
# 冻结编码器，只训练新加的分类层（体现“先无监督学表示，再加标签”）
for p in ae.encoder.parameters():  # 遍历编码器中的所有参数
    p.requires_grad = False  # 关闭梯度，冻结编码器，使其在监督阶段不再更新
model = LabeledModel(ae.encoder).to(device)  # 用冻结后的编码器构建分类模型并放到设备
clf_opt = torch.optim.Adam(model.classifier.parameters(), lr=1e-2)  # 只优化分类头参数
ce_loss_fn = nn.CrossEntropyLoss()  # 分类任务使用交叉熵损失
# ---------------------------------------------------------------------------
@torch.no_grad()  # 评估时不需要计算梯度，节省计算
def accuracy(net, x, y):  # 定义准确率计算函数
    net.eval()  # 切换到评估模式（关闭 dropout 等训练行为）
    pred = net(x.to(device)).argmax(dim=1).cpu()  # 得到预测类别编号并转回 CPU
    return (pred == y).float().mean().item()  # 计算预测正确比例并返回 Python 数值
# ---------------------------------------------------------------------------
print("\n=== 阶段 2：添加标签后微调分类头 ===")  # 打印阶段 2 标题
print(f"仅使用 {n_labeled} 条带标签样本")  # 提示当前使用了多少条标签数据
model.train()  # 将分类模型切换到训练模式
for epoch in range(1, 51):  # 监督微调 50 个 epoch
    total = 0.0  # 累计当前 epoch 的总交叉熵损失
    for batch_x, batch_y in labeled_loader:  # 按批取出带标签样本
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 特征和标签都放到设备上
        logits = model(batch_x)  # 前向传播得到未归一化的类别分数
        loss = ce_loss_fn(logits, batch_y)  # 计算预测与真实标签的交叉熵
        clf_opt.zero_grad()  # 清空分类头上一轮梯度
        loss.backward()  # 反向传播，主要更新分类头参数
        clf_opt.step()  # 根据梯度更新分类头
        total += loss.item() * batch_x.size(0)  # 按样本数加权累计损失
    if epoch % 10 == 0:  # 每 10 个 epoch 打印训练与测试表现
        train_acc = accuracy(model, x_train[:n_labeled], y_train[:n_labeled])  # 计算有标签训练集准确率
        test_acc = accuracy(model, x_test, y_test)  # 计算测试集准确率
        print(  # 打印当前 epoch 的损失和准确率
            f"epoch {epoch:02d}  ce_loss={total / n_labeled:.4f}  "  # 平均交叉熵损失
            f"labeled_acc={train_acc:.3f}  test_acc={test_acc:.3f}"  # 有标签准确率与测试准确率
        )  # print 语句结束
print("\n完成：无监督表示学习 -> 加分类头用标签预测类别")  # 打印程序结束提示
