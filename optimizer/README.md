# Optimizer

[TOC]

## api

引入 `torch.optim`


## Model 建立

```python
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)   # hidden layer
        self.predict = torch.nn.Linear(20, 1)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x


nets = {
    "net_SGD" : Model(),
    "net_momentum" : Model(),
    "net_adam" : Model(),
    "net_adagrad" : Model(),
    "net_adadelta": Model(),
}
```

## Optimizer 建立

```python
optimizers={
    "SGD" : optim.SGD(nets["net_SGD"].parameters(),lr = LR),
    "Momentum" : optim.SGD(nets["net_momentum"].parameters(),lr=LR,momentum=0.8),
    "Adam" : optim.Adam(nets["net_adam"].parameters(),lr=LR,betas=(0.9,0.99)),
    "Adagrad" : optim.Adagrad(nets["net_adagrad"].parameters(),lr=LR),
    "Adadelta": optim.Adadelta(nets["net_adadelta"].parameters(),lr=1)
}
```

## loss 使用 MSELOSS()


```python
loss_func = torch.nn.MSELoss()
loss_history = [[] for _ in range(len(nets))] #存放 loss
```


## Train

```python
for epoch in range(EPOCH+1):
    if epoch%10==0:
        print("Epoch: ",epoch)
    


    for step ,(b_x,b_y)in enumerate(data_loader):
        for model , opti ,l_h in zip(nets.values(),optimizers.values(),loss_history):
            
            output = model(b_x)
            loss = loss_func(output,b_y)
            opti.zero_grad()
            loss.backward()
            opti.step()
            l_h.append(loss.data.numpy())
```

## 繪圖

```python
labels = [o for o in optimizers.keys()]



for i,l_h in enumerate(loss_history):
    plt.plot(l_h,label=labels[i],linewidth=1)
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0, 0.2))
plt.show()
```

![](https://i.imgur.com/pHAaMju.png)
