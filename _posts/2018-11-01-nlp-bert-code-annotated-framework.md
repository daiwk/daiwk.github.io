---
layout: post
category: "nlp"
title: "bert代码解读——framework"
tags: [bert代码解读, bert code, framework]
---

目录

<!-- TOC -->

- [modeling.py](#modelingpy)
  - [公共函数](#%E5%85%AC%E5%85%B1%E5%87%BD%E6%95%B0)
    - [assert-rank](#assert-rank)
    - [get-shape-list](#get-shape-list)
    - [create-initializer](#create-initializer)
    - [dropout](#dropout)
    - [layer-norm](#layer-norm)
    - [layer-norm-and-dropout](#layer-norm-and-dropout)
    - [embedding-lookup](#embedding-lookup)
    - [embedding-postprocessor](#embedding-postprocessor)
  - [BertConfig](#bertconfig)
    - [BertConfig初始化](#bertconfig%E5%88%9D%E5%A7%8B%E5%8C%96)
    - [BertConfig方法](#bertconfig%E6%96%B9%E6%B3%95)
    - [from-dict(classmethod)](#from-dictclassmethod)
    - [from-json-file(classmethod)](#from-json-fileclassmethod)
    - [to-dict](#to-dict)
    - [to-json-string](#to-json-string)
  - [BertModel](#bertmodel)
    - [初始化](#%E5%88%9D%E5%A7%8B%E5%8C%96)
    - [get-pooled-output](#get-pooled-output)
    - [get-sequence-output](#get-sequence-output)
    - [get-all-encoder-layers](#get-all-encoder-layers)
    - [get-embedding-output](#get-embedding-output)
    - [get-embedding-table](#get-embedding-table)
- [extract-features.py](#extract-featurespy)
- [optimization.py](#optimizationpy)
- [tokenization.py](#tokenizationpy)

<!-- /TOC -->

**注：由于标题不能有下划线，所以把函数名/文件名里的_换成了-**

## modeling.py

高仿[https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/transformer_layers.py#L99](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/transformer_layers.py#L99)的transformer_encoder部分。

### 公共函数

#### assert-rank

注意：

tensor的rank表示一个tensor**需要的索引数目**来唯一表示任何一个元素。也就是通常所说的 “order”, “degree”或”ndims”，不是矩阵的秩。。参考[https://blog.csdn.net/lenbow/article/details/52152766](https://blog.csdn.net/lenbow/article/details/52152766)

例如：

```python
#’t’ is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
# shape of tensor ‘t’ is [2, 2, 3]
rank(t) = 3
```

函数的功能：如果输入tensor的rank和预期的不一样，就抛异常

参数：

+ tensor：输入的tf.Tensor
+ expected_rank：Python integer or list of integers，期望的rank
+ name：error message中的tensor的名字

```python
def assert_rank(tensor, expected_rank, name=None):
  if name is None:
    name = tensor.name

  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = tf.get_variable_scope().name
    raise ValueError(
        "For the tensor `%s` in scope `%s`, the actual rank "
        "`%d` (shape = %s) is not equal to the expected rank `%s`" %
        (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))
```

#### get-shape-list

参数：

+ tensor：一个需要返回shape的tf.Tensor
+ expected_rank：int。输入tensor期望的rank，如果输入tensor的rank不等于这个数，会抛异常

```python
def get_shape_list(tensor, expected_rank=None, name=None):
  if name is None:
    name = tensor.name

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape
```

#### create-initializer

对```tf.truncated_normal_initializer```的简单封装

```python
def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.truncated_normal_initializer(stddev=initializer_range)
```

#### dropout

```python
def dropout(input_tensor, dropout_prob):
  """Perform dropout.

  Args:
    input_tensor: float Tensor.
    dropout_prob: Python float. The probability of dropping out a value (NOT of
      *keeping* a dimension as in `tf.nn.dropout`).

  Returns:
    A version of `input_tensor` with dropout applied.
  """
  if dropout_prob is None or dropout_prob == 0.0:
    return input_tensor

  output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
  return output
```

#### layer-norm

```python
def layer_norm(input_tensor, name=None):
  """Run layer normalization on the last dimension of the tensor."""
  return tf.contrib.layers.layer_norm(
      inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)
```

#### layer-norm-and-dropout

```python
def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
  """Runs layer normalization followed by dropout."""
  output_tensor = layer_norm(input_tensor, name)
  output_tensor = dropout(output_tensor, dropout_prob)
  return output_tensor
```

#### embedding-lookup

返回一个shape是```[batch_size, seq_length, embedding_size]```的tensor，还有shape为```[vocab_size, embedding_size]```的整个embedding_table

参数：

+ input_ids：shape为包含了word ids的```[batch_size, seq_length]```的tensor
+ vocab_size：embedding vocabulary的size
+ embedding_size：word embeddings的width
+ initializer_range：Embedding初始化的range
+ word_embedding_name：embedding table的名字
+ use_one_hot_embeddings：true: 使用one-hot的embedding；false：使用```tf.nn.embedding_lookup()```，如下所述，tpu用one-hot好，cpu/gpu用非one-hot好

```python
def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings",
                     use_one_hot_embeddings=False):
  ## 此函数假设输入的shape是[batch_size, seq_length, num_inputs]。如果是[batch_size, seq_length]，会reshape成[batch_size, seq_length, 1]
  if input_ids.shape.ndims == 2:
    ## tf.expand_dims在axis处插入维度1进入一个tensor中
    input_ids = tf.expand_dims(input_ids, axis=[-1])

  ## shape是[vocab_size, embedding_size]的embedding table
  embedding_table = tf.get_variable(
      name=word_embedding_name,
      shape=[vocab_size, embedding_size],
      initializer=create_initializer(initializer_range))

  if use_one_hot_embeddings:
    ## 把[batch_size, seq_length, 1]的input_ids变成[batch_size*seq_length]的一个tensor
    flat_input_ids = tf.reshape(input_ids, [-1])
    ## 变成一个[batch_size*seq_length, vocab_size]的one-hot的tensor，depth参数的含义就是vocab_size
    one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
    ## [batch_size*seq_length, vocab_size]的one_hot_input_ids和[vocab_size, embedding_size]的embedding_table矩阵相乘，得到[batch_size*seq_length,embedding_size]的output
    output = tf.matmul(one_hot_input_ids, embedding_table)
  else:
    ## [batch_size, seq_length, 1]的input_ids去[vocab_size, embedding_size]的embedding_table中lookup，得到一个[batch_size, seq_length, 1, embedding_size]的output
    output = tf.nn.embedding_lookup(embedding_table, input_ids)

  input_shape = get_shape_list(input_ids)

  ## reshape成[batch_size, seq_length, 1 * embedding_size]的输出
  output = tf.reshape(output,
                      input_shape[0:-1] + [input_shape[-1] * embedding_size])
  return (output, embedding_table)
```

#### embedding-postprocessor

参数：

+ input_tensor：shape是```[batch_size, seq_length, embedding_size]```的float Tensor
+ use_token_type：是否要为embedding加上```token_type_ids```
+ token_type_ids：shape是```[batch_size, seq_length]```的int32 Tensor
+ token_type_vocab_size：```token_type_ids```的vocabulary size
+ token_type_embedding_name：```token_type_ids```的embedding_table的名字，默认"token_type_embeddings"
+ use_position_embeddings：是否要为序列里每个token的位置加上position embeddings
+ position_embedding_name：position_embeddings的embedding_table的名字，默认"position_embeddings"
+ initializer_range：权重初始化的range
+ max_position_embeddings：最大的sequence length，可以比输入的sequence length长，但不能比它短
+ dropout_prob：**最终输出tensor**的dropout rate

```python
def embedding_postprocessor(input_tensor,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1):
  ## input_shape是[batch_size, seq_length, embedding_size]
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  width = input_shape[2]

  if seq_length > max_position_embeddings:
    raise ValueError("The seq length (%d) cannot be greater than "
                     "`max_position_embeddings` (%d)" %
                     (seq_length, max_position_embeddings))

  output = input_tensor

  if use_token_type:
    if token_type_ids is None:
      raise ValueError("`token_type_ids` must be specified if"
                       "`use_token_type` is True.")
    ## 搞一个[token_type_vocab_size, embedding_size]的token_type的embedding_table
    token_type_table = tf.get_variable(
        name=token_type_embedding_name,
        shape=[token_type_vocab_size, width],
        initializer=create_initializer(initializer_range))
    # 由于token_type的vocab很小，所以直接用one-hot，这样能更快，这个用法和上面的embedding_lookup函数一样，不再赘述
    flat_token_type_ids = tf.reshape(token_type_ids, [-1])
    one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
    token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
    token_type_embeddings = tf.reshape(token_type_embeddings,
                                       [batch_size, seq_length, width])
    output += token_type_embeddings

  if use_position_embeddings:
    full_position_embeddings = tf.get_variable(
        name=position_embedding_name,
        shape=[max_position_embeddings, width],
        initializer=create_initializer(initializer_range))
    # position embedding table是一个learned variable，对[0, 1, 2, ..., max_position_embeddings-1]来讲，
    # full_position_embeddings的shape就是[max_position_embeddings, width]
    #
    # 而当前的序列长度是seq_length，所以针对[0, 1, 2, ... seq_length-1], 可以对full_position_embeddings做个slice
    # 传入给slice的begin是[0,0]，size是[seq_length,-1]，所以是对输入的shape取[0:seq_len, 0:-1]，所以
    # slice的结果position_embeddings的shape是[seq_length, width]
    if seq_length < max_position_embeddings:
      position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                     [seq_length, -1])
    else:
      position_embeddings = full_position_embeddings
    # output.shape是[batch_size, seq_length, embedding_size]，num_dims是3
    num_dims = len(output.shape.as_list())

    ## 其实这里就是把position_broadcast_shape写成[1, seq_length, width]
    position_broadcast_shape = []
    for _ in range(num_dims - 2):
      position_broadcast_shape.append(1)
    position_broadcast_shape.extend([seq_length, width])
    # 把position_embeddings从[seq_length, width]给reshape成[1, seq_length, width]，方便和output相加
    # 第一维是1，是因为batch里的每一条数据，相同position加的position embedding是一样的
    position_embeddings = tf.reshape(position_embeddings,
                                     position_broadcast_shape)
    output += position_embeddings

  output = layer_norm_and_dropout(output, dropout_prob)
  return output
```

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

#### from-dict(classmethod)

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

#### from-json-file(classmethod)

```python
  @classmethod
  def from_json_file(cls, json_file):
    """Constructs a `BertConfig` from a json file of parameters."""
    with tf.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))
```

#### to-dict

关于deepcopy，参考[https://daiwk.github.io/posts/knowledge-python.html#copy-deepcopy](https://daiwk.github.io/posts/knowledge-python.html#copy-deepcopy)

```python
  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output
```

#### to-json-string

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

+ config：```BertConfig```instance.
+ is_training：true: training model；false：eval model。用于控制是否dropout。
+ input_ids：shape是```[batch_size, seq_length]```的int32 Tensor。
+ input_mask：shape是```[batch_size, seq_length]```的int32 Tensor。
+ token_type_ids：shape是```[batch_size, seq_length]```的int32 Tensor。
+ use_one_hot_embeddings：使用one-hot embedding，还是```tf.embedding_lookup()```。TPU上设成True会更快，cpu/gpu上设成False更快。
+ scope：variable scope，默认是```bert```。

实现分成以下几步：

首先是input_mask/token_type_ids/batch_size/seq_length的确定：

```python
    config = copy.deepcopy(config)
    if not is_training:
      config.hidden_dropout_prob = 0.0
      config.attention_probs_dropout_prob = 0.0
    # 期望input_ids的shape是两维，即[batch_size, seq_length]
    input_shape = get_shape_list(input_ids, expected_rank=2)
    batch_size = input_shape[0]
    seq_length = input_shape[1]

    if input_mask is None:
      ## 默认input_mask全是1
      input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

    if token_type_ids is None:
      ## 默认token_type_ids全是0
      token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)
```

然后确定网络结构：

```python
    with tf.variable_scope("bert", scope):
      with tf.variable_scope("embeddings"):
        # 对输入的word ids进行emb
        (self.embedding_output, self.embedding_table) = embedding_lookup(
            input_ids=input_ids,
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            initializer_range=config.initializer_range,
            word_embedding_name="word_embeddings",
            use_one_hot_embeddings=use_one_hot_embeddings)
        # 对wordid的emb结果，加上type embed和position emb，然后normalize并dropout输出
        self.embedding_output = embedding_postprocessor(
            input_tensor=self.embedding_output,
            use_token_type=True,
            token_type_ids=token_type_ids,
            token_type_vocab_size=config.type_vocab_size,
            token_type_embedding_name="token_type_embeddings",
            use_position_embeddings=True,
            position_embedding_name="position_embeddings",
            initializer_range=config.initializer_range,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob)

      with tf.variable_scope("encoder"):
        # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
        # mask of shape [batch_size, seq_length, seq_length] which is used
        # for the attention scores.
        attention_mask = create_attention_mask_from_input_mask(
            input_ids, input_mask)

        # Run the stacked transformer.
        # `sequence_output` shape = [batch_size, seq_length, hidden_size].
        self.all_encoder_layers = transformer_model(
            input_tensor=self.embedding_output,
            attention_mask=attention_mask,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            intermediate_act_fn=get_activation(config.hidden_act),
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            initializer_range=config.initializer_range,
            do_return_all_layers=True)

      self.sequence_output = self.all_encoder_layers[-1]
      # The "pooler" converts the encoded sequence tensor of shape
      # [batch_size, seq_length, hidden_size] to a tensor of shape
      # [batch_size, hidden_size]. This is necessary for segment-level
      # (or segment-pair-level) classification tasks where we need a fixed
      # dimensional representation of the segment.
      with tf.variable_scope("pooler"):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. We assume that this has been pre-trained
        first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
        self.pooled_output = tf.layers.dense(
            first_token_tensor,
            config.hidden_size,
            activation=tf.tanh,
            kernel_initializer=create_initializer(config.initializer_range))

```

#### get-pooled-output

返回pooled_output

```python
  def get_pooled_output(self):
    return self.pooled_output
```

#### get-sequence-output

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

#### get-all-encoder-layers

返回all_encoder_layers

```python
  def get_all_encoder_layers(self):
    return self.all_encoder_layers
```

#### get-embedding-output

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

#### get-embedding-table

返回embedding_table

```python
  def get_embedding_table(self):
    return self.embedding_table
```

## extract-features.py

## optimization.py

## tokenization.py
