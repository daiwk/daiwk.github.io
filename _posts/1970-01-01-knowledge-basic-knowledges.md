---
layout: post
category: "knowledge"
title: "基础知识"
tags: [基础知识, json]
---

目录

<!-- TOC -->

- [json+protobuf的坑](#jsonprotobuf%e7%9a%84%e5%9d%91)

<!-- /TOC -->

## json+protobuf的坑

protobuf中的bool类型，如果通过python的json来给：

```python
js["aa"] = 1
```

那pb读出来会是false。。变成0

所以应该

```python
js["aa"] = True
```
