# Sequential


## 使用

```python
model = torch.nn.Sequential(
    torch.nn.Linear(1,10),
    torch.nn.ReLU(),  # 跟 nn.functional 不一樣 這個是 class
    torch.nn.Linear(10,1)
)

print(model)
```

```bash
Sequential (
  (0): Linear (1 -> 10)
  (1): ReLU ()
  (2): Linear (10 -> 1)
)
```

`torch.nn.ReLU()` 與 `torch.nn.functional.relu()` 不一樣的，前者是 class 後者是 function