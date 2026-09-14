# PyTorch：无监督 → 加标签

示例脚本：`unsupervised_then_label.py`

1. **无监督**：自编码器重建数据，学习特征（不用标签）
2. **加标签**：冻结编码器，接分类头，用少量标签做分类

```bash
pip install torch
python unsupervised_then_label.py
```

参考课程（正向/反向传播）：  
https://www.bilibili.com/video/BV1vV4y1K717/
