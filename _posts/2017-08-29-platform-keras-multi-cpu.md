---
layout: post
category: "platform"
title: "keras使用多核cpu"
tags: [keras, theano, 多核cpu, htop]
---

## 开启theano后端

vim ``` ~/.keras/keras.json ```，修改为
```
{
    "epsilon": 1e-07,
    "floatx": "float32",
    "image_data_format": "channels_last",
    "backend": "theano"
}
```

## 打开openmp的flag

```python
import theano
theano.config.openmp = True
```

## 确保gcc版本

至少要是482的

```
export PATH=/opt/compiler/gcc-4.8.2/bin/:$PATH
```
## 设置环境变量并运行

```
OMP_NUM_THREADS=20 python xxx.py
```

## 用htop查看进程情况

```
jumbo install htop
```

htop样例如下：

![](../assets/htop.png)
