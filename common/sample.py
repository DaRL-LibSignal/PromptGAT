import numpy as np
import matplotlib.pyplot as plt

def custom_normal_distribution(x, num_samples=1000):
    # 确保输入x在范围内，将其映射到0.25，0.5，0.75或0.99
    x = min(max(x, 0.25), 0.99)

    # 构建正态分布
    mean = x
    std_dev = 0.1  # 调整标准差来控制分布的宽度
    samples = np.random.normal(loc=mean, scale=std_dev, size=num_samples)

    # 离散化采样结果
    samples = np.clip(samples, 0.25, 0.99)
    samples = np.round(samples * 4) / 4  # 四舍五入到0.25，0.5，0.75或0.99

    return samples

# 测试并可视化结果
x_input = 0.5
samples = custom_normal_distribution(x_input)

# 绘制直方图
plt.hist(samples, bins=[0.25, 0.5, 0.75, 0.99, 1.0], align='left', rwidth=0.8)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.xticks([0.25, 0.5, 0.75, 0.99])
plt.title('Custom Normal Distribution with Mean = {}'.format(x_input))
plt.show()