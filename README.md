# A Multi Directional Approach for Accelerating Single-Node Image Classification Neural Network Training via Pruning

## _Train and test_
- Install requirements:
```
$ pip install -r requirements . txt
```

- Run code:
```
$ python main.py --config /path/to/a/single/config.json
```
```
$ python main.py --batch /path/to/directory/of/multiple/configs/
```


## _References_
[**ADMM**](https://arxiv.org/abs/1804.03294) implementation based on _[KaiqiZhang's TensorFlow implementation](https://github.com/KaiqiZhang/admm-pruning)_
and _[bzantium's PyTorch implementation](https://github.com/bzantium/pytorch-admm-pruning)_.
