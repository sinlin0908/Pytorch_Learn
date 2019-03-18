# RNN Classification

## 資料

同 CNN

## RNN 架構

```python
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM,self).__init__()
        
        '''
        batch_first -> True  (batch,time_step,input_size)
                    -> False (time_step,batch,input_size)
        '''
        self.lstm_layer = nn.LSTM(
            input_size = config["INPUT_SIZE"],
            hidden_size = 64,
            num_layers = 1, 
            batch_first=True,
        )
        
        self.output_layer = nn.Linear(
            in_features=64,
            out_features=10
        )
        
    def forward(self,x):
        '''
        LSTM:
            Input: 
                 input:
                 h_0 : initial hidden state for each element, default 0
                 c_0 : initial cell state for each element, default 0
                 
            Output:
                output: output features (h_t) from the "last layer" of the LSTM,for each time
                h_n: hidden state for t = seq_len.
                c_n: the cell state for t = seq_len
        '''
        lstm_output,(h_n,h_c) = self.lstm_layer(x,None)
        
        out = self.output_layer(lstm_output[:,-1,:]) # get last time step
                                                     # (batch,time step,input)
        return out
```

## Train

```python
def draw_loss(loss_history):
    
    plt.plot(loss_history,label="loss",linewidth=1)
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim((0, 4))
    plt.show()
    

def train(model,data_loader):
    l_h = []
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(),lr=config["LR"])
    loss_fun = nn.CrossEntropyLoss()
    
    for epoch in range(config['epoch']):
        for step,(b_x,b_y) in enumerate(data_loader):
            
            b_x,b_y = b_x.view(-1,28,28).to(device), b_y.to(device) # reshape x to (batch, time_step, input_size)

             
            optimizer.zero_grad()      
            
            output = model(b_x)
            
            loss = loss_fun(output,b_y)
            
            
            
            loss.backward()
            optimizer.step()
            
            l_h.append(loss.item()) 

            if step % 200 == 0:
                
                print('Epoch: ', epoch, '| step ',step,'| train loss: %.4f' % loss.item())
                
                
            
    draw_loss(l_h)

```
![](https://i.imgur.com/j4OmHZV.png)