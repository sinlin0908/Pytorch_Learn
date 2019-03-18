# AutoEncoder



用 mnist 手寫辨識進行


## 設定

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

import os


torch.manual_seed(1)
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


config = {
    "epoch":10,
    "batch_size":64,
    "LR":0.001,
    "DOWNLOAD_MNIST": False if os.path.exists('../mnist') else True,
    "N_TEST_IMG":5
}
```

## 資料集


```python
'''
不需要 test data
'''
train_data = torchvision.datasets.MNIST(
    root='../mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=config["DOWNLOAD_MNIST"],
)

data_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=config["batch_size"],
    shuffle=True
)
```

因為是非監督式學習，所以不需要 test data


## AutoEncoder 架構

```python
class AutoEncoder(nn.Module):
    def __init__(self):
        
        super(AutoEncoder,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(
                in_features=28*28,
                out_features=128,
            ),
#             nn.Tanh(),
            nn.ReLU(),
            nn.Linear(
                in_features=128,
                out_features=64,
            ),
#             nn.Tanh(),
            nn.ReLU(),
            nn.Linear(
                in_features=64,
                out_features=12,
            ),
#             nn.Tanh(),
            nn.ReLU(),
            nn.Linear(
                in_features=12,
                out_features=3,
            ),
        )
        
        
        self.decoder = nn.Sequential(
            nn.Linear(
                in_features=3,
                out_features=12,
            ),
#             nn.Tanh(),
            nn.ReLU(),
            nn.Linear(
                in_features=12,
                out_features=64,
            ),
#             nn.Tanh(),
            nn.ReLU(),
            nn.Linear(
                in_features=64,
                out_features=128,
            ),
#             nn.Tanh(),
            nn.ReLU(),
            nn.Linear(
                in_features=128,
                out_features=28*28,
            ),
            nn.Sigmoid(),  #因為原始資料為 0~1的數值
            
        )
        
        
    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return encoded,decoded
```
使用 ReLu 效果好像比較好

## Train

```python
def draw_loss(loss_history):
    
    plt.plot(loss_history,label="loss",linewidth=1)
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim((0, 4))
    plt.show()
    

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


    
def train(model,data_loader,view_data):
    l_h = []
    
    model.train()
    
    
    optimizer = optim.Adam(model.parameters(),lr=config['LR'])
    loss_fun = nn.MSELoss()
    
    for epoch in range(config["epoch"]):
        for step ,(b_x,b_label) in enumerate(data_loader):
            
            # x y 都是從 x 來
            b_x,b_y = b_x.view(-1,28*28).to(device) , b_x.view(-1,28*28).to(device)
            
            optimizer.zero_grad()
            
            encoded,decoded = model(b_x)
            
            loss = loss_fun(decoded,b_y)
            
            loss.backward()
            
            optimizer.step()
            
            l_h.append(loss.item())
            
            if step % 500 == 0:
                
                print('Epoch: ', epoch, '| step ',step,'| train loss: %.4f' % loss.item())
                
                pic = to_img(view_data.detach())
                save_image(pic, './img2/image_source.png')
                
                _, out = model(view_data.to(device))
                
                pic = to_img(out.cpu().detach())
                save_image(pic, './img2/image_{}_{}.png'.format(epoch,step))
    
    draw_loss(l_h)
    
    
view_data = train_data.train_data[:config['N_TEST_IMG']].view(-1, 28*28).type(torch.FloatTensor)/255.

train(model,data_loader,view_data)
```

訓練時每500step進行觀察
資料使用train dataset前5張

原圖
![](https://i.imgur.com/DoI2npR.png)

0 epoch , 0 step
![](https://i.imgur.com/KupUC12.png)

1 epoch , 500 step
![](https://i.imgur.com/bAH22C7.png)

10 epoch , 0 step
![](https://i.imgur.com/isrK3Cu.png)

50 epoch , 500 step
![](https://i.imgur.com/3u3O5jT.png)

95 epoch, 0 step
![](https://i.imgur.com/jzVbgiL.png)

99 epoch , 500 step
![](https://i.imgur.com/oLqvJmh.png)

5 和 3，4 和 9 有點分不清楚的樣子

比較具有對稱的數字容易答對，像是 0 和 1

# Test

```python
def test(model,test_data):
    model.eval()
    pic = to_img(test_data.detach())       
    save_image(pic, './img2/test_source.png')
                
    _, out = model(test_data.to(device))

    pic = to_img(out.cpu().detach())    
    save_image(pic, './img2/test.png')
    

        
test_data = train_data.train_data[-config['N_TEST_IMG']:].view(-1, 28*28).type(torch.FloatTensor)/255.        

test(model,test_data)
```

使用 tain dataset 的後5張圖做測試

原圖
![](https://i.imgur.com/UsZlXh6.png)

結果
![](https://i.imgur.com/OOXZvBV.png)
