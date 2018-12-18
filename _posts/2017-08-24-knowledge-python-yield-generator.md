---
layout: post
category: "knowledge"
title: "python generator"
tags: [generator, yield]
---

目录

<!-- TOC -->


<!-- /TOC -->
[http://www.cnblogs.com/cotyb/p/5260032.html](http://www.cnblogs.com/cotyb/p/5260032.html)

groupby经典用法(前提是输入的数据相同key的结果已经相邻)：

```python
import itertools
def get_group_key(line):
    segs = line.rstrip('\n').split('\t')
    return segs[0] 

for key, iter in itertools.groupby(sys.stdin, get_group_key):
    if key == "":
        continue
    for line in iter:
        val = line.rstrip('\n').split('\t')
```