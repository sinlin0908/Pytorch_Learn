# Pytorch Variable


[TOC]

## Api

- torch.autograd 提供
- 參數 requires_grad=True 為反向傳播時會計算這個節點的梯度


```python
import torch
from torch.autograd import Variable

tensor = torch.FloatTensor([[1,2],[3,4]])

variable = Variable(tensor,requires_grad=True) # Variable


# x^2
tensor_out = torch.mean(tensor*tensor)

variable_out = torch.mean(variable*variable)

print(
    "\ntensor out: ",tensor_out,
    "\n\nvariable out: ",variable_out
)
```

```bash

tensor out:  7.5 

variable out:  Variable containing:
 7.5000
[torch.FloatTensor of size 1]

```

## 反向傳播

```python
variable_out.backward() #誤差反向傳遞

print(
    "gradient",
    variable.grad
)

```

```bash
gradient Variable containing:
 0.5000  1.0000
 1.5000  2.0000
[torch.FloatTensor of size 2x2]
```

補充 gradient 算法

$Out = \dfrac{1}{4} * sum(variable^2)$ 

$Gredient = \dfrac{\mathrm{d}out}{\mathrm{d}variable} =\dfrac{1}{4} * 2 * variable$

## Variable 轉換 

```python
variable.data  # variable -> Tensor

variable.data.numpy() # variable -> numpy
```