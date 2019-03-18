# Save Model

## 用法

```python
# 保存整個 model
def save(model,step):
    torch.save(model,"model_{}.pkl".format(step))

def restore_model(file_name):
    return torch.load(file_name)
    
# 只保存 model 參數
def save_parameters(model,step):
    torch.save(model.state_dict(),"model_para_{}.pkl".format(step))

def restore_model_paras(file_name):
    
    model = Net(1,10,1)
    model.load_state_dict( torch.load_state_dict(file_name))
```