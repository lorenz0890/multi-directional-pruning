# A Multi Directional Approach for Accelerating Single-Node Image Classification Neural Network Training via Pruning

Implementation of the methods proposed in the [Master's Thesis](https://utheses.univie.ac.at/detail/63874#) of the same title.

## Abstract
Deep neural networks (DNNs) continuously increase in architectural complexity and DNN applications are under pressure to handle an every-increasing amount of data as well as more and more complex problems. This has created demand for approaches reducing time and space complexity of such networks during training and inference, especially for training under resource constraints or inference under time constraints.} State-of-the-Art (SotA) works aimed at reducing DNNs' demand for memory and computation time during training and/or the inference can be categorized into techniques applying pruning, quantization, neural architecture search (NAS), distributed learning or a combination of these to the network. In this work, the focus lies on intra-training DNN pruning (i.e., pruning the network during and with the aim of accelerating training) for which new approaches are introduced. This is accomplished by an extensive literature survey and in-depth analysis of the State-of-the-Art in the field of intra training DNN pruning, highlighting current scientific gaps and synthesizing and implementing novel solutions to a selected subset of the identified open research questions. The proposed solutions in this thesis will be subjected to extensive analytical and empirical evaluation in comparison with other State-of-the-Art methods in order to demonstrate their scientific contribution.

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
