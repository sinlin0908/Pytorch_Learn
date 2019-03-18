# Activation Function

[toc]

## Api

需要引入 `torch.nn.functional`


```python
import torch
import torch.nn.functional 
from torch.autograd import Variable


def relu(x):
    return torch.nn.functional.relu(x)

def sigmoid(x):
    return torch.nn.functional.sigmoid(x)

def tanh(x):
    return torch.nn.functional.tanh(x)

def softmax(x):
    return torch.nn.functional.softmax(x)
```

