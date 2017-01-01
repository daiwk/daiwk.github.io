---
layout: post
category: "other"
title: "bigflow的坑们"
tags: [bigflow,]
---


自动缩减："abaci.dag.datasize.per.reduce": "20000000", # 20m

default_concurrency：在create_pileline的时候设置

**自动缩减只能调小不能调大。**所以如果default concurrency比较小，就不会调了。这个default concurrcency最好是设置成比预估稍微大一点的并发，如果设置的太大，会影响dce shuffle的性能。

也就是说

```python
split_size = total_size / datasize_per_reduce

tasks = 0
if split_size > default_concurrency:
    tasks = default_concurrency
else:
    tasks = split_size
 
```
