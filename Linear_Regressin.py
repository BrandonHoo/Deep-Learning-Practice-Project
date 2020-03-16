'''简单的说： 线性回归对于输入x与输出y有一个映射f，y=f(x),而f的形式为aX+b。
   其中a和b是两个可调的参数，我们训练的时候就是训练a，b这两个参数。
'''  
# 注意，这里我们使用了一个新库叫 seaborn 如果报错找不到包的话请使用pip install seaborn 来进行安装
import torch
from torch.nn import Linear, Module, MSELoss        #线性模型，损失函数
from torch.optim import SGD                         #优化器，SGD
import numpy as np  
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

#定义一个线性函数y=a*x+b   a,b=5,7
x = np.linspace(0,20,500)
y = 5*x + 7
# plt.plot(x,y) 画出图型

#生成一些随机的点，来作为我们的训练数据
x = np.random.rand(256)
noise = np.random.randn(256) / 4
y = x * 5 + 7 + noise
df = pd.DataFrame()
df['x'] = x
df['y'] = y

#图上显示下我们生成的数据
#sns.lmplot(x='x', y='y', data=df);

#使用PyTorch建立一个线性的模型来对其进行拟合，这就是所说的训练的过程，由于只有一层线性模型，
#其中参数(1, 1)代表输入输出的特征(feature)数量都是1. Linear 模型的表达式是 y=w*x+b，其中 w代表权重， b代表偏置
model=Linear(1, 1)

# 损失函数我们使用均方损失函数：MSELoss
criterion = MSELoss()

#优化器我们选择最常见的优化方法 SGD，就是每一次迭代计算 mini-batch 的梯度，然后对参数进行更新，学习率 0.01
optim = SGD(model.parameters(), lr = 0.01)

#训练3000次
epochs = 30000

'''

准备训练数据: x_train, y_train 的形状是 (256, 1)， 
代表 mini-batch 大小为256， feature 为1.
astype('float32') 是为了下一步可以直接转换为 torch.float
'''
x_train = x.reshape(-1, 1).astype('float32')
y_train = y.reshape(-1, 1).astype('float32')

#开始训练了
for i in range(epochs):
    # 整理输入和输出的数据，这里输入和输出一定要是torch的Tensor类型
    inputs = torch.from_numpy(x_train)  #from_numpy 将numpy转换为torch
    labels = torch.from_numpy(y_train)
    #使用模型进行预测
    outputs = model(inputs)
    #梯度置0，否则会累加
    optim.zero_grad()
    # 计算损失
    loss = criterion(outputs, labels)
    # 反向传播
    loss.backward()
    # 使用优化器默认方法优化
    optim.step()
    if (i%100==0):
        #每 100次打印一下损失函数，看看效果
        print('epoch {}, loss {:1.4f}'.format(i,loss.data.item()))

    #item() 以列表返回可遍历（键-值）元组的数组。语法dict.item()

#用 model.parameters() 提取模型参数。 w，b是我们所需要训练的模型参数 我们期望的数据 w=5，b=7$可以做一下对比
[w, b] = model.parameters()
print (w.item(),b.item())

#可视化一下我们的模型，看看我们训练的数据，如果你不喜欢seaborn，可以直接使用matplot
predicted = model.forward(torch.from_numpy(x_train)).data.numpy()
plt.plot(x_train, y_train, 'go', label = 'data', alpha = 0.3)
plt.plot(x_train, predicted, label = 'predicted', alpha = 1)
plt.legend()
plt.show()


#计算偏差

print (5-w.data.item(),7-b.data.item())