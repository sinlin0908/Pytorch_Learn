# Dataset

[TOC]

## 用途 

整理數據結構，使用他來包裝自己的數據，進行批次訓練


## Api

引入 `torch.utils.data`

有以下功能：

1. TensorDataset

    包裝數據和目標張量的數據集
    
```python
class torch.utils.data.TensorDataset(data_tensor, target_tensor)
```

- data_tensor (Tensor) :　包含样本数据
- target_tensor (Tensor) :　包含样本目标（标签）

2. DataLoader

    組合數據集和採樣器，並在數據集上提供單進程或多進程迭代器

```python
class torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        sampler=None, 
        num_workers=0, 
        collate_fn=<function default_collate>, 
        pin_memory=False, 
        drop_last=False
)
```

- dataset (Dataset) : 加載數據的數據集
- batch_size (int): 每個 batch 加載多少個樣本 (默認: 1)
- shuffle (bool) : 設置為 True 時會在每個epoch重新打亂數據 (默認: False)
- sampler (Sampler) : 定義從數據集中提取樣本的策略。如果指定，則忽略 shuffle 參數
- num_workers (int) : 用多少個子進程加載數據。 0表示數據將在主進程中加載(默認: 0)
- drop_last (bool) : 如果數據集大小不能被batch size整除，則設置為True後可刪除最後一個不完整的batch。如果設為False並且數據集的大小不能被batch size整除，則最後一個batch將更小。 (默認: False)

## 範例


```python
import torch
import torch.utils.data as Data

BATCH_SIZE = 5
EPOCH = 3

# Data

x = torch.linspace(1,10,10) # torch tensor
y = torch.linspace(10,1,10) # torch tensor

torch_dataset = Data.TensorDataset(data_tensor=x,target_tensor=y)

# Data loader

loader = Data.DataLoader(
    dataset = torch_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True,
    num_workers = 2,
)

# display

for e in range(EPOCH):
    for step, (batch_x,batch_y) in enumerate(loader):
        print(
            "Epoch: ",e,
            "  Step: ",step,
            "  batch_x: ",batch_x.numpy(),
            "  batch_y: ",batch_y.numpy()
        )

```
```bash
Epoch:  0   Step:  0   batch_x:  [ 3.  6.  5. 10.  7.]   batch_y:  [8. 5. 6. 1. 4.]
Epoch:  0   Step:  1   batch_x:  [9. 4. 1. 2. 8.]   batch_y:  [ 2.  7. 10.  9.  3.]
Epoch:  1   Step:  0   batch_x:  [1. 5. 8. 4. 6.]   batch_y:  [10.  6.  3.  7.  5.]
Epoch:  1   Step:  1   batch_x:  [ 3. 10.  2.  9.  7.]   batch_y:  [8. 1. 9. 2. 4.]
Epoch:  2   Step:  0   batch_x:  [7. 3. 8. 1. 2.]   batch_y:  [ 4.  8.  3. 10.  9.]
Epoch:  2   Step:  1   batch_x:  [ 6. 10.  4.  5.  9.]   batch_y:  [5. 1. 7. 6. 2.]
```

一個 epoch 有兩個 step (10/5)


