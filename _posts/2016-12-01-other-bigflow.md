---
layout: post
category: "other"
title: "bigflow的坑们"
tags: [bigflow,]
---

目录

<!-- TOC -->

- [自动缩减](#自动缩减)
- [一些reduce的task非常慢](#一些reduce的task非常慢)
- [数据太长](#数据太长)
- [lazyvar](#lazyvar)

<!-- /TOC -->

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


### lazyvar

[http://bigflow.baidu.com/doc/faq.html#bigflow](http://bigflow.baidu.com/doc/faq.html#bigflow)

目前外部词典的加载方式主要有3种：

+ 方法1. 如果词典文件极小（不超过10M）直接使用Python的闭包功能

+ 方法2. 如果词典文件较大（10M < 词典大小 < 1G），可以使用bigflow提供的lazy_var模块在远端加载词典。

lazy_var从1.0.4版本开始提供，目前只提供 一个临时下载地址：wget http://bigflow.baidu.com/download/module/lazy_var.py:

```python
# coding: utf-8
from bigflow import base, lazy_var

"""
使用建议：
    以下demo中的my_lazy_var是一个全局变量，实际上my_lazy_var可以在全局和局部使用。

    我们建议最好在局部使用。

    如果多个函数依赖my_lazy_var，可以适当修改业务代码，封装在类中，
    my_lazy_var作为类的一个成员，在类的所有成员函数中以self.my_var_lazy使用，
    效果几乎等价于全局.
"""

"""
mydict.txt

key1    value1
key2    value2
key3    value3
"""

def load_dict(path):
    data = dict()
    with open(path, "r") as f:
        for line in f:
            (key, value) = line.split()
            data[key] = value
    return data

my_lazy_var = lazy_var.declare(lambda: load_dict("./mydict.txt"))

def get_value(key):
    # 获取lazy_var内容
    my_dict = my_lazy_var.get()
    return my_dict.get(key)

def main():
    my_dict = my_lazy_var.get()
    # 可以在本地直接获取lazy_var内容
    # ["value1", "value2", "value3"]
    print my_dict.values()

    pipeline = base.Pipeline.create("local")
    pipeline.add_file("./mydict.txt", "./mydict.txt")
    keys = pipeline.parallelize(["key1", "key2", "key3", "key4"])
    # 可以在transforms中获取lazy_var内容
    values = keys.map(get_value)
    # ["value1", "value2", "value3", None]
    print values.get()

if __name__ == "__main__":
    main()

```

+ 方法3. 使用side_input（在1.0.3之前的版本中此功能在下游并发较大时，会出现执行效率比较差的情况）

另外，如果自定义的类型(python的class)会出现bad marshal等错误时，可以用lazyvar来搞

```
import my_lazy_var
from brand_tagger import BrandTagger

def init_tagger():
    """
    """

    tagger = BrandTagger()
    fname_model = "./third/model/model"
    fname_bigram = "./third/model/bigram.dat"
    tagger.set_fnames(fname_model, fname_bigram)
    return tagger


def do_tag(line, tagger):
    """
    """

    if not tagger.has_loaded():
        ret = tagger.load_model()

    csid = line[0]
    userid = line[1]
    real_rawpicurl = line[2]
    encoded_data = line[3]
    raw_width = line[4]
    raw_height = line[5]
    descs = line[6]
    picurl = line[7]
    srcs = line[8]
    ext_json = line[9]
    to_cmp_descs_str = line[10]
    search_word = line[11]
    search_word_list = search_word.split(";")

    formatted_res_list = []
    for item in search_word_list:
        sub_item = item.split(":")
        word = sub_item[0]
        prob = sub_item[1]
        brands = tagger.do_tag(word)
        brand_tag_res = word
        for brand in brands:
            brand_tag_res = brand_tag_res.replace(brand, "{brand}")
        formatted_res_list.append(":".join([brand_tag_res, prob]))
    formatted_res = ";".join(formatted_res_list)

    res_list = line[: -1] + [formatted_res]

    return res_list

_pipeline.add_file(workspace_path + "./opt-feeds-image/image-tagging/postprocess/my_lazy_var.py", "my_lazy_var.py")
tagger = my_lazy_var.declare(init_tagger)

format_brand_res = filter_res\
        .map(lambda x: do_tag(x, tagger.get()))\
        .map(lambda x: "\t".join(x))


```