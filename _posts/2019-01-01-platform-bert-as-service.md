---
layout: post
category: "platform"
title: "bert as service"
tags: [bert as service, ]

---

目录

<!-- TOC -->

- [基本使用](#%E5%9F%BA%E6%9C%AC%E4%BD%BF%E7%94%A8)
  - [安装：](#%E5%AE%89%E8%A3%85)
  - [启动server](#%E5%90%AF%E5%8A%A8server)
  - [启动client](#%E5%90%AF%E5%8A%A8client)

<!-- /TOC -->

[https://github.com/hanxiao/bert-as-service](https://github.com/hanxiao/bert-as-service)

官网：[https://bert-as-service.readthedocs.io/en/latest/](https://bert-as-service.readthedocs.io/en/latest/)

## 基本使用

### 安装：

```shell
pip install -U bert-serving-server bert-serving-client
```

要求：

+ server的py要>=3.5,tf要>=1.10
+ client可以是py2或者py3

另外需要把预训练好的模型下下来~

### 启动server

```shell
$workspace/bin/python3.6 \
        $workspace/bin/bert-serving-start \
        -model_dir=$workspace/../chinese_L-12_H-768_A-12/ \
        -num_worker=4
```

### 启动client

```python
from bert_serving.client import BertClient
remote_ip = 'xx.xx.xx.xx'
bc = BertClient(ip=remote_ip)  
print(bc.encode(['First do it', 'then do it right', 'then do it better']))
```

