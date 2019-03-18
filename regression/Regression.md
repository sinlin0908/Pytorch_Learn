# Regression

[TOC]

實戰線性預測

## 創資料

```python
import torch
from torch.autograd import Variable
import torch.nn.functional as Function

import matplotlib.pyplot as plt

config = {
    'epoch':500,
    'lr': 0.001
}


# Data

class Data(object):
    def __init__(self):
        # unsqueeze 一維轉成二維
        # [1,2,3] -> [[1,2,3]]
        # (3) -> (3,1)
        self.x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)

        self.y = self.x.pow(2) + 0.2*torch.rand(self.x.size())

        self.x , self.y = Variable(self.x) , Variable(self.y)

    def get_data(self):
        return self.x , self.y
```

資料呈現

```python
def draw(x,y):
    plt.scatter(x,y)
    plt.show()

# print Data
data = Data()

x,y = data.get_data()

draw(x.data.numpy(),y.data.numpy())
```

![](https://i.imgur.com/YgIDjvE.png)

## 定義神經網路

```python
class Net(torch.nn.Module):
    def __init__(self,n_feature=1,n_hidden=1,n_label=1):
        super(Net,self).__init__()

        # Define Network
        # 這樣不代表已經建好神經網路

        self.hidden_layer = torch.nn.Linear(n_feature,n_hidden)
        self.output_layer = torch.nn.Linear(n_hidden,n_label)


    def forward(self,x):
        x = Function.relu(self.hidden_layer(x))
        x = self.output_layer(x)

        return x
```

## 開始建立

```python

net = Net(n_feature=1,n_hidden=10,n_label=1)

print(net) # 查看結構
```

```bash
Net (
  (hidden_layer): Linear (1 -> 10)
  (output_layer): Linear (10 -> 1)
)
```

OS : pytorch 這個部份真是太好用了 XD

## Training

```python
# 給 optimizer 全部的參數 -> net.parameters()

optimizer = torch.optim.Adam(net.parameters(),lr=config['lr'])
loss_func = torch.nn.MSELoss()

net.train() # 設置 model 為 trainning mode

for step in range(config['epoch']):
    out = net(x)
    loss = loss_func(out,y)

    if step % 50 == 0:
        print("Epoch[{}/{}] loss:{}".format(step,config['epoch'],loss.data.numpy()))

    optimizer.zero_grad() # 初始梯度，上一次保留的梯度刪除
    loss.backward()
    optimizer.step() # 優化
```

Loss 值

```bash
Epoch[0/500] loss:[0.00353637]
Epoch[50/500] loss:[0.00337658]
Epoch[100/500] loss:[0.00335121]
Epoch[150/500] loss:[0.00334545]
Epoch[200/500] loss:[0.00334114]
Epoch[250/500] loss:[0.0033383]
Epoch[300/500] loss:[0.00333618]
Epoch[350/500] loss:[0.00333437]
Epoch[400/500] loss:[0.0033331]
Epoch[450/500] loss:[0.00333163]
```

## Evaluation

```python
net.eval() # evaluation model
predict = net(x)
predict = predict.data.numpy()

draw(x.data.numpy(),predict)
```

![](https://i.imgur.com/xJM8Trl.png)

比較

```python
plt.plot(x.data.numpy(), y.data.numpy(), 'bo', label='Original data')
plt.plot(x.data.numpy(), predict, color='red',linewidth=5,label='Fitting Line')
plt.legend()
plt.show()
```

![](https://i.imgur.com/XU2wcDb.png)

## Note

> tainning 時要指定 model 為 tain mode
> evaluation 時要指定 model 為 evaluation model

兩個方法是針對在網絡訓練和測試時採用不同方式的情況，比如 Batch Normalization 和 Dropout。
