# A Multi Directional Approach for Accelerating Single-Node Image Classification Neural Network Training via Pruning

Implementation of the methods proposed in the [Master's Thesis](https://utheses.univie.ac.at/detail/63874#) of the same title.


## _Train and test_
- Install requirements:
```
$ pip install -r requirements.txt
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
