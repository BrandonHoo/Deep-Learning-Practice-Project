import torch     # pytorch 实现
import torch.nn as nn 
import numpy as np   #处理矩阵运算

'''
1 logistic回归会在线性回归后再加一层logistic函数的调用，主要
用于二分类预测

2 使用UCI German Credit 数据集 
  german.data-numeric 是已经使用numpy 处理好的数值化数据，
  可以直接numpy 调用

'''
#读取数据
data=np.loadtxt("german.data-numeric") #将数据放到文件中加载

#对数据做归一化处理
n,l=data.shape    #shape 返回矩阵大小
print(l-1)
for i in range(l-1):      #按列索引
    meanVal=np.mean(data[:,i])  #求均值  [:,i] 取所有行第i列所有值
    stdVal=np.std(data[:i])     # 标准差
    data[:,i]=(data[:,i]-meanVal)/stdVal

#打乱数据
np.random.shuffle(data)

'''
区分数据集和测试集：
区分规则：900条用于训练，100条用于测试
前24列为24个维度，最后一个要打的标签（0，1）
'''
train_data=data[:900,l-1]
#train_data=train_data.view(len(train_data),1)
print(train_data.shape)
train_lab=data[:900,l-1]-1
test_data=data[:900,l-1]
test_lab=data[:900,l-1]-1

#定义模型
class Logistic_Regression(nn.Module):
    #初始化模型
    def __init__(self):
        # super(Logistic_Regression,self) 首先找到 Logistic_Regression的父类（就是类nn.Module）
        # 然后把类 Logistic_Regression的对象转换为类 nn.Module的对象
        super(Logistic_Regression,self).__init__()
        self.fc=nn.Linear(900,2)    # 输入通道为900，输出通道为2

    #前向传播
    def forward(self,x):
        out=self.fc(x)
        out=torch.sigmoid(out)  # sigmoid 激活
        return out


#测试集上的准确率
def test(pred,lab):
    t=pred.max(-1)[1]==lab 
    return torch.mean(t.float())

#设置
net=Logistic_Regression()
criterion=nn.CrossEntropyLoss()   #定义损失函数
optm=torch.optim.Adam(net.parameters()) #利用Adam 进行优化
epochs=1000  #训练次数


for i in range(epochs):
    #指定模型为训练模式，计算梯度
    net.train()
    #将numpy 输入转换为torch的Tensor
    x=torch.from_numpy(train_data).float()
    print(x)
    y=torch.from_numpy(train_lab).long()
    print(y.shape)
    y_hat=net(x) #x 为训练数据
    loss=criterion(y_hat,y) #计算损失
    optm.zero_grad()        #前一步损失清零
    loss.backward()         #f=反向传播
    optm.step()             #优化
    if (i+1)%100==0:        #每一百100输出相关信息
        net.eval()
        test_in=torch.from_numpy(test_data).float()
        test_l=torch.from_numpy(test_lab).long()
        test_out=net(test_in)
        accu=test(test_out,test_l)
        print("Epoch:{},Loss:{:.4f},Accuracy:{:.2f}",format(i+1,loss.item(),accu))
