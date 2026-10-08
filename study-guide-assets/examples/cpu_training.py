"""独立教学例：纯 Python / CPU，固定玩具图像特征，不是原 Lab 0 答案。
运行：python3 cpu_training.py
类别0=横线，类别1=竖线；这里只训练2维特征到2类的线性分类头。
"""
import math

# 每项为：人为设计的[横向响应, 纵向响应]，以及类别标签。
DATA = [([1.0, 0.0], 0), ([0.8, 0.2], 0),
        ([0.0, 1.0], 1), ([0.2, 0.8], 1)]
W = [[0.0, 0.0], [0.0, 0.0]]  # [输出类别][输入特征]
b = [0.0, 0.0]
LR = 0.5

def forward(x):
    logits = [sum(w * v for w, v in zip(row, x)) + bias
              for row, bias in zip(W, b)]
    exp = [math.exp(z - max(logits)) for z in logits]
    return [v / sum(exp) for v in exp]

def evaluate():
    probs = [forward(x) for x, _ in DATA]
    loss = sum(-math.log(p[y]) for p, (_, y) in zip(probs, DATA)) / len(DATA)
    correct = sum(max(range(2), key=lambda j: p[j]) == y
                  for p, (_, y) in zip(probs, DATA))
    return loss, correct / len(DATA)

print('初始 loss=%.6f, 玩具训练集 accuracy=%.2f' % evaluate())
for step in range(1, 21):
    # 每个优化步骤清空梯度；此处是整个4样本batch的平均梯度。
    dW = [[0.0, 0.0], [0.0, 0.0]]
    db = [0.0, 0.0]
    for x, y in DATA:
        p = forward(x)
        for c in range(2):
            dz = (p[c] - float(c == y)) / len(DATA)  # softmax+交叉熵梯度
            db[c] += dz
            for j in range(2):
                dW[c][j] += dz * x[j]
    if step == 1:
        print('第1步平均 dW=', [[round(v, 6) for v in row] for row in dW])
    for c in range(2):
        b[c] -= LR * db[c]
        for j in range(2):
            W[c][j] -= LR * dW[c][j]
    if step in (1, 5, 20):
        print('更新%02d loss=%.6f, 玩具训练集 accuracy=%.2f' % (step, *evaluate()))
print('x=[1,0] 的概率=', [round(p, 6) for p in forward([1.0, 0.0])])
print('注意：这些是固定玩具训练数据上的结果，不是泛化性能。')
