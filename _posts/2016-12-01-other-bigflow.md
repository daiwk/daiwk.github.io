---
layout: post
category: "other"
title: "bigflow的坑们"
tags: [bigflow,]
---


### 自动缩减

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

### 一些reduce的task非常慢

如果后面还有执行慢的问题的话。可以设置下cpu_profile，这样那里计算耗时可以通过pprof显示出来

```
pipeline = base.Pipeline.create(*****, cpu_profile=True)
```


### 数据太长

看dce-writer-xxxxxxxxxxxxxxx文件，出现这句就挂了。。。

```
FATAL [writer.cc:188] 0113 17:27:47.340183 1342 | CHECK failed: key.length() + value.length() < max_length: Too Big Data.Crashing...
```