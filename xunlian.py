import sys
import caffe
import numpy as np

# 设置Caffe的运行模式
caffe.set_mode_gpu()  # 使用GPU
# caffe.set_mode_cpu()  # 使用CPU

# 加载模型和权重
model_def = '"./mobilenet_train.prototxt"'
model_weights = './MobileNetSSD_80000.caffemodel'  # 替换为实际权重文件路径
。
# 创建网络
net = caffe.Net(model_def, model_weights, caffe.TRAIN)

# 设置学习率和其他超参数
base_lr = 0.0005
max_iter = 80000
snapshot_prefix = 'snapshot/mobilenet'

# 训练循环
for iter in range(max_iter):
    # 进行前向传播和反向传播
    net.forward()
    net.backward()

    # 更新权重
    for layer_name, layer in net.layers.items():
        if layer_name in net.params:
            for param in net.params[layer_name]:
                param.data -= base_lr * param.diff  # 简单的SGD更新

    # 每1000次保存一次快照
    if iter % 1000 == 0:
        net.save(snapshot_prefix + f'_iter_{iter}.caffemodel')
        print(f"Snapshot saved at iteration {iter}")

print("Training complete.")