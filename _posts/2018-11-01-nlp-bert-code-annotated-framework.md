---
layout: post
category: "nlp"
title: "bert代码解读——framework"
tags: [bert代码解读, bert code, framework]
---

目录

<!-- TOC -->

- [modeling.py](#modelingpy)
  - [BertConfig](#bertconfig)
    - [BertConfig初始化](#bertconfig%E5%88%9D%E5%A7%8B%E5%8C%96)
    - [BertConfig方法](#bertconfig%E6%96%B9%E6%B3%95)
    - [from_dict(classmethod)](#fromdictclassmethod)
    - [from_json_file(classmethod)](#fromjsonfileclassmethod)
    - [to_dict](#todict)
    - [to_json_string](#tojsonstring)
  - [BertModel](#bertmodel)
    - [初始化](#%E5%88%9D%E5%A7%8B%E5%8C%96)
    - [get_pooled_output](#getpooledoutput)
    - [get_sequence_output](#getsequenceoutput)
    - [get_all_encoder_layers](#getallencoderlayers)
    - [get_embedding_output](#getembeddingoutput)
    - [get_embedding_table](#getembeddingtable)
- [extract_features.py](#extractfeaturespy)
- [optimization.py](#optimizationpy)
- [tokenization.py](#tokenizationpy)

<!-- /TOC -->

## modeling.py

高仿[https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/transformer_layers.py#L99](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/transformer_layers.py#L99)的transformer_encoder部分。

### BertConfig

#### BertConfig初始化

```python
class BertConfig(object):
  """Configuration for `BertModel`."""

  def __init__(self,
               vocab_size,
               hidden_size=768,
               num_hidden_layers=12,
               num_attention_heads=12,
               intermediate_size=3072,
               hidden_act="gelu",
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               max_position_embeddings=512,
               type_vocab_size=16,
               initializer_range=0.02):
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.type_vocab_size = type_vocab_size
    self.initializer_range = initializer_range
```

参数如下：

+ vocab_size：```inputs_ids```的vocabulary size
+ hidden_size：encoder layers和pooler layer的size
+ num_hidden_layers：Transformer encoder的hidden layer数
+ num_attention_heads：Transformer encoder的每个attention layer的attention heads数
+ intermediate_size：Transformer encoder的"intermediate" layer(例如feed-forward)的size
+ hidden_act：encoder and pooler的激活函数
+ hidden_dropout_prob：embeddings, encoder, 和pooler的所有全连接的dropout rate
+ attention_probs_dropout_prob：attention probabilities的dropout rate
+ max_position_embeddings：最大的sequence长度，通常设得比较大（512 or 1024 or 2048）
+ type_vocab_size：```token_type_ids```的vocabulary size
+ initializer_range：所有权重矩阵的truncated_normal_initializer的stdev

#### BertConfig方法

+ classmethod

注：```classmethod```修饰符对应的函数**不需要实例化**，**不需要self参数**，但第一个参数需要是表示自身类的cls参数，可以来**调用类的属性，类的方法，实例化对象等**。

例如：

```python
class A(object):
    bar = 1
    def func1(self):  
        print ('foo') 
    @classmethod
    def func2(cls):
        print ('func2')
        print (cls.bar)
        cls().func1()   # 调用 foo 方法
 
A.func2()               # 不需要实例化
```

+ ```__dict__```变量

实例的__dict__仅存储与该实例相关的实例属性，

```python
>>> x = BertConfig(vocab_size=None)
>>> print x.__dict__
{'type_vocab_size': 16, 'vocab_size': None, 'num_attention_heads': 12, 'num_hidden_layers': 12, 'attention_probs_dropout_prob': 0.1, 'max_position_embeddings': 512, 'initializer_range': 0.02, 'hidden_act': 'gelu', 'hidden_size': 768, 'intermediate_size': 3072, 'hidden_dropout_prob': 0.1}
```

类的__dict__存储所有实例共享的变量和函数(类属性，方法等)，类的__dict__并不包含其父类的属性。

```python
class A(object):
    def __init__(self, a):
        self.a = a
    def func1(self, xx):
        self.xx = xx

    @classmethod
    def func2(cls, mm):
        return mm

ccc = A(a=3)
print ccc.__dict__
print A.__dict__

# 输出

{'a': 3}
{'func2': <classmethod object at 0x100812478>, '__module__': '__main__', 'func1': <function func1 at 0x1007fc758>, '__dict__': <attribute '__dict__' of 'A' objects>, '__weakref__': <attribute '__weakref__' of 'A' objects>, '__doc__': None, '__init__': <function __init__ at 0x1007fc140>}

```

看看BertConfig的方法们：

#### from_dict(classmethod)

```python
  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `BertConfig` from a Python dictionary of parameters."""
    config = BertConfig(vocab_size=None)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    return config
```

其中的```six.iteritems```函数如下：

```python
    def iteritems(d, **kw):
        return iter(d.items(**kw))
```

使用如下：

```python
>>> a={"a":3, "b":9}
>>> for i in six.iteritems(a):
...     print i
...
('a', 3)
('b', 9)
```

#### from_json_file(classmethod)

```python
  @classmethod
  def from_json_file(cls, json_file):
    """Constructs a `BertConfig` from a json file of parameters."""
    with tf.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))
```

#### to_dict

注：

Python中的对象之间赋值(=运算)时是按引用传递的，如果需要拷贝对象，需要使用标准库中的copy模块。

copy.copy 浅拷贝 vs copy.deepcopy 深拷贝：

+ 对于简单的 object，用 shallow copy 和 deep copy 没区别

```python
>>> import copy
>>> origin = 1
>>> cop1 = copy.copy(origin) 
#cop1 是 origin 的shallow copy
>>> cop2 = copy.deepcopy(origin) 
#cop2 是 origin 的 deep copy
>>> origin = 2
>>> origin
2
>>> cop1
1
>>> cop2
1
#cop1 和 cop2 都不会随着 origin 改变自己的值
>>> cop1 == cop2
True
>>> cop1 is cop2
True
```

+ 复杂的object， 如list中套着list的情况，**shallow copy中的子list，并未从原object真的「独立」出来。**

```python
>>> import copy
>>> origin = [1, 2, [3, 4]]
#origin 里边有三个元素：1， 2，[3, 4]
>>> cop1 = copy.copy(origin)
>>> cop2 = copy.deepcopy(origin)
>>> cop1 == cop2
True
>>> cop1 is cop2
False 
#cop1 和 cop2 看上去相同，但已不再是同一个object
>>> origin[2][0] = "hey!" 
>>> origin
[1, 2, ['hey!', 4]]
>>> cop1
[1, 2, ['hey!', 4]]
>>> cop2
[1, 2, [3, 4]]
#把origin内的子list [3, 4] 改掉了一个元素，观察 cop1 和 cop2
```

浅copy容易遇到的『坑』：

```python
# 把 [1, 2, 3] 看成一个物品。a = [1, 2, 3] 就相当于给这个物品上贴上 a 这个标签。而 b = a 就是给这个物品又贴上了一个 b 的标签。
>>> a = [1, 2, 3]
>>> b = a
>>> a = [4, 5, 6] //赋新的值给 a
>>> a
[4, 5, 6]
>>> b
[1, 2, 3]
# a 的值改变后，b 并没有随着 a 变。
# a = [4, 5, 6] 就相当于把 a 标签从 [1 ,2, 3] 上撕下来，贴到了 [4, 5, 6] 上。
# 在这个过程中，[1, 2, 3] 这个物品并没有消失。 b 自始至终都好好的贴在 [1, 2, 3] 上，既然这个 reference 也没有改变过。 b 的值自然不变。

>>> a = [1, 2, 3]
>>> b = a
>>> a[0], a[1], a[2] = 4, 5, 6 //改变原来 list 中的元素
>>> a
[4, 5, 6]
>>> b
[4, 5, 6]
# a 的值改变后，b 随着 a 变了
# a[0], a[1], a[2] = 4, 5, 6 则是直接改变了 [1, 2, 3] 这个物品本身。把它内部的每一部分都重新改装了一下。内部改装完毕后，[1, 2, 3] 本身变成了 [4, 5, 6]。
# 而在此过程当中，a 和 b 都没有动，他们还贴在那个物品上。因此自然 a b 的值都变成了 [4, 5, 6]。
```

```python
  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output
```

#### to_json_string

```python
  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
```

### BertModel

#### 初始化

```python
class BertModel(object):
  def __init__(self,
               config,
               is_training,
               input_ids,
               input_mask=None,
               token_type_ids=None,
               use_one_hot_embeddings=True,
               scope=None):
```

参数如下：

+ config
+ is_training
+ input_ids
+ input_mask
+ token_type_ids
+ use_one_hot_embeddings
+ scope

#### get_pooled_output

返回pooled_output

```python
  def get_pooled_output(self):
    return self.pooled_output
```

#### get_sequence_output

返回encoder的最后一个隐层

```python
  def get_sequence_output(self):
    """Gets final hidden layer of encoder.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the final hidden of the transformer encoder.
    """
    return self.sequence_output
```

#### get_all_encoder_layers

返回all_encoder_layers

```python
  def get_all_encoder_layers(self):
    return self.all_encoder_layers
```

#### get_embedding_output

返回embedding_output，shape是```[batch_size, seq_length, hidden_size]```，是加好了positional embeddings和token type embeddings，然后过了layer norm的结果，即transformer的input。

```python
  def get_embedding_output(self):
    """Gets output of the embedding lookup (i.e., input to the transformer).

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the output of the embedding layer, after summing the word
      embeddings with the positional embeddings and the token type embeddings,
      then performing layer normalization. This is the input to the transformer.
    """
    return self.embedding_output
```

#### get_embedding_table

返回embedding_table

```python
  def get_embedding_table(self):
    return self.embedding_table
```

## extract_features.py

## optimization.py

## tokenization.py
