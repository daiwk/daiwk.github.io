---
layout: post
category: "nlp"
title: "bert代码解读——application"
tags: [bert代码解读, bert code, application]
---

目录

<!-- TOC -->

- [basics](#basics)
  - [TPUEstimator](#tpuestimator)
    - [train](#train)
    - [predict](#predict)
    - [evaluate](#evaluate)
- [extract-features](#extract-features)
  - [extract-features.py](#extract-featurespy)
    - [InputExample](#inputexample)
    - [InputFeatures](#inputfeatures)
    - [-truncate-seq-pair](#truncate-seq-pair)
    - [read-examples](#read-examples)
    - [convert-examples-to-features](#convert-examples-to-features)
    - [input-fn-builder](#input-fn-builder)
    - [model-fn-builder](#model-fn-builder)
    - [main](#main)
- [pretrain](#pretrain)
  - [create-pretraining-data.py](#create-pretraining-datapy)
  - [run-pretraining.py](#run-pretrainingpy)
- [classification](#classification)
  - [run-classifier.py](#run-classifierpy)
- [squad](#squad)
  - [run-squad.py](#run-squadpy)

<!-- /TOC -->

## basics

### TPUEstimator

类的定义在```site-packages/tensorflow/contrib/tpu/python/tpu/tpu_estimator.py```中

#### train

在```tensorflow/tensorflow/contrib/tpu/python/tpu/tpu_estimator.py```文件中（如果装的是cpu/gpu版的，好像没这个函数）：

```python
  def train(self,
            input_fn,
            hooks=None,
            steps=None,
            max_steps=None,
            saving_listeners=None):
    rendezvous = error_handling.ErrorRendezvous(num_sources=3)
    self._rendezvous[model_fn_lib.ModeKeys.TRAIN] = rendezvous
    try:
      return super(TPUEstimator, self).train(
          input_fn=input_fn, hooks=hooks, steps=steps, max_steps=max_steps,
          saving_listeners=saving_listeners
      )
    except Exception:  # pylint: disable=broad-except
      rendezvous.record_error('training_loop', sys.exc_info())
    finally:
      rendezvous.record_done('training_loop')
      rendezvous.raise_errors()
```

#### predict

在```tensorflow/tensorflow/contrib/tpu/python/tpu/tpu_estimator.py```文件中（如果装的是cpu/gpu版的，好像没这个函数）：


```python
  def predict(self,
              input_fn,
              predict_keys=None,
              hooks=None,
              checkpoint_path=None,
              yield_single_examples=True):
    rendezvous = error_handling.ErrorRendezvous(num_sources=3)
    self._rendezvous[model_fn_lib.ModeKeys.PREDICT] = rendezvous
    try:
      for result in super(TPUEstimator, self).predict(
          input_fn=input_fn,
          predict_keys=predict_keys,
          hooks=hooks,
          checkpoint_path=checkpoint_path,
          yield_single_examples=yield_single_examples):
        yield result
    except Exception:  # pylint: disable=broad-except
      rendezvous.record_error('prediction_loop', sys.exc_info())
    finally:
      rendezvous.record_done('prediction_loop')
      rendezvous.raise_errors()

    rendezvous.record_done('prediction_loop')
    rendezvous.raise_errors()
```

#### evaluate

看tf的源码，在```tensorflow/tensorflow/contrib/tpu/python/tpu/tpu_estimator.py```文件中（如果装的是cpu/gpu版的，好像没这个函数）：

```python
  def evaluate(self, input_fn, steps=None, hooks=None, checkpoint_path=None,
               name=None):
    rendezvous = error_handling.ErrorRendezvous(num_sources=3)
    self._rendezvous[model_fn_lib.ModeKeys.EVAL] = rendezvous
    try:
      return super(TPUEstimator, self).evaluate(
          input_fn, steps=steps, hooks=hooks, checkpoint_path=checkpoint_path,
          name=name
      )
    except Exception:  # pylint: disable=broad-except
      rendezvous.record_error('evaluation_loop', sys.exc_info())
    finally:
      rendezvous.record_done('evaluation_loop')
      rendezvous.raise_errors()
```

## extract-features

### extract-features.py

#### InputExample

```python
class InputExample(object):

  def __init__(self, unique_id, text_a, text_b):
    self.unique_id = unique_id
    self.text_a = text_a
    self.text_b = text_b
```

#### InputFeatures

```python
class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
    self.unique_id = unique_id
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.input_type_ids = input_type_ids
```

#### -truncate-seq-pair

```python
def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""
  # 保证tokens_a+tokens_b的总长度小于等于max_length
  # 如果不满足，把比较长的那个list的最后一个元素删了，然后循环，直到满足为止
  # 当一个句子很短时，这样做与对每个句子删掉相同比例的token要更make sense，
  # 因为短句子中的token信息量应该会比长句子更大
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop() # 把tokens_a的最后一个元素删了
    else:
      tokens_b.pop() # 把tokens_b的最后一个元素删了
```

#### read-examples

```python
def read_examples(input_file):
  """Read a list of `InputExample`s from an input file."""
  examples = []
  unique_id = 0
  with tf.gfile.GFile(input_file, "r") as reader:
    while True:
      # 将每行转成unicode
      line = tokenization.convert_to_unicode(reader.readline())
      if not line:
        break
      line = line.strip()
      text_a = None
      text_b = None
      # 以『|||』进行分隔，前面是句子A，后面是句子B
      m = re.match(r"^(.*) \|\|\| (.*)$", line)
      if m is None:
        text_a = line
      else:
        text_a = m.group(1)
        text_b = m.group(2)
      # 使用InputExample类封装一下
      examples.append( 
          InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
      unique_id += 1
  return examples
```

#### convert-examples-to-features

```python
def convert_examples_to_features(examples, seq_length, tokenizer):
  """Loads a data file into a list of `InputFeatures`s."""

  features = []
  for (ex_index, example) in enumerate(examples):
    # 对句子a进行分词
    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    # 对句子b进行分词
    if example.text_b:
      tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
      # 对tokens_a和tokens_b进行裁剪，保证总长度不大于seq_length - 3
      # -3是因为有[CLS], [SEP], [SEP]
      _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
    else:
      # -2是因为没有tokens_b的时候，只有[CLS], [SEP]
      if len(tokens_a) > seq_length - 2:
        tokens_a = tokens_a[0:(seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    # 其中，"type_ids"表示是第一句(0)还是第二句(1)
    #
    # 对于classification tasks, [CLS]的向量可以被当做是"sentence vector". 当然，只有当整个model已经fine-tuned的时候，这才make sense
    tokens = []
    input_type_ids = []
    tokens.append("[CLS]")
    input_type_ids.append(0)
    for token in tokens_a:
      tokens.append(token)
      input_type_ids.append(0)
    tokens.append("[SEP]")
    input_type_ids.append(0)

    if tokens_b:
      for token in tokens_b:
        tokens.append(token)
        input_type_ids.append(1)
      tokens.append("[SEP]")
      input_type_ids.append(1)
    # 将token转成id（在tokenizer中，读vocab文件，行号就是其id，所以不能简单地增量训）
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # 因为seq_len可能比实际输入的序列长，所以需要padding
    # 实际输入的mask是1
    input_mask = [1] * len(input_ids)

    # 比实际输入长，到seq_length的部分，用0进行padding，mask也写成0
    # 注意，vocab文件中，第0行，也就是第0个token，是[PAD]，专门用来padding的
    while len(input_ids) < seq_length:
      input_ids.append(0)
      input_mask.append(0)
      input_type_ids.append(0)

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length

    if ex_index < 5: # ex_index是第几个输入example，只有前5个example打这个日志
      tf.logging.info("*** Example ***")
      tf.logging.info("unique_id: %s" % (example.unique_id))
      tf.logging.info("tokens: %s" % " ".join(
          [tokenization.printable_text(x) for x in tokens]))
      tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
      tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
      tf.logging.info(
          "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

    features.append(
        InputFeatures(
            unique_id=example.unique_id,
            tokens=tokens,
            input_ids=input_ids,
            input_mask=input_mask,
            input_type_ids=input_type_ids))
  return features
```

#### input-fn-builder

```python
def input_fn_builder(features, seq_length):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_unique_ids = []
  all_input_ids = []
  all_input_mask = []
  all_input_type_ids = []

  for feature in features:
    all_unique_ids.append(feature.unique_id)
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_input_type_ids.append(feature.input_type_ids)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "unique_ids":
            tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_type_ids":
            tf.constant(
                all_input_type_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
    })

    d = d.batch(batch_size=batch_size, drop_remainder=False)
    return d

  return input_fn
```

#### model-fn-builder

```python
def model_fn_builder(bert_config, init_checkpoint, layer_indexes, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    # 这里的features是input_fn_builder的输出，格式如上
    unique_ids = features["unique_ids"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    input_type_ids = features["input_type_ids"]

    model = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=input_type_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    if mode != tf.estimator.ModeKeys.PREDICT:
      raise ValueError("Only PREDICT modes are supported: %s" % (mode))

    tvars = tf.trainable_variables()
    scaffold_fn = None

    # initialized_variable_names: 有哪些变量在checkpoint中已经初始化了
    (assignment_map,
     initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
         tvars, init_checkpoint)
    if use_tpu:

      def tpu_scaffold():
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        return tf.train.Scaffold()

      scaffold_fn = tpu_scaffold
    else:
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        # 这些变量在checkpoint中已经init了
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    # 是一个list，每个元素的shape是[batch_size, seq_length, hidden_size]
    all_layers = model.get_all_encoder_layers()

    predictions = {
        "unique_id": unique_ids,
    }

    ## 对于需要输出的indexes，从all_layers里取出来
    for (i, layer_index) in enumerate(layer_indexes):
      predictions["layer_output_%d" % i] = all_layers[layer_index]

    output_spec = tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn
```

#### main

首先，定义如下几个变量：

```python
  tf.logging.set_verbosity(tf.logging.INFO)
  # 期望输出的layer_index们，例如: -1,-2,-3
  layer_indexes = [int(x) for x in FLAGS.layers.split(",")]

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
  # 切词类
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  # tpu run_config
  run_config = tf.contrib.tpu.RunConfig(
      master=FLAGS.master,
      tpu_config=tf.contrib.tpu.TPUConfig(
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))
```

然后

```python
  # 读文件
  examples = read_examples(FLAGS.input_file)
  # 切词，并保证句子a+句子b再加上padding和[CLS]/[SEP]等的总长度不大于max_seq_length
  # 把unique_id/oken/input_ids/mask/input_type_ids存在features中
  features = convert_examples_to_features(
      examples=examples, seq_length=FLAGS.max_seq_length, tokenizer=tokenizer)
  # unique_id就是输入样本的行号，把每行对应的具体feature存到dict里
  unique_id_to_feature = {}
  for feature in features:
    unique_id_to_feature[feature.unique_id] = feature
```

然后

```python
  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      layer_indexes=layer_indexes,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_one_hot_embeddings)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      predict_batch_size=FLAGS.batch_size)

  input_fn = input_fn_builder(
      features=features, seq_length=FLAGS.max_seq_length)
```

然后

```python
  with codecs.getwriter("utf-8")(tf.gfile.Open(FLAGS.output_file,
                                               "w")) as writer:
    for result in estimator.predict(input_fn, yield_single_examples=True):
      unique_id = int(result["unique_id"])
      feature = unique_id_to_feature[unique_id]
      output_json = collections.OrderedDict()
      output_json["linex_index"] = unique_id # 第几个样本
      all_features = []
      for (i, token) in enumerate(feature.tokens):
        all_layers = []
        for (j, layer_index) in enumerate(layer_indexes):
          layer_output = result["layer_output_%d" % j]
          layers = collections.OrderedDict()
          layers["index"] = layer_index
          layers["values"] = [
              round(float(x), 6) for x in layer_output[i:(i + 1)].flat
          ]
          all_layers.append(layers)
        features = collections.OrderedDict()
        features["token"] = token
        features["layers"] = all_layers
        all_features.append(features)
      output_json["features"] = all_features
      writer.write(json.dumps(output_json) + "\n")
```


## pretrain

### create-pretraining-data.py

### run-pretraining.py

## classification

### run-classifier.py

## squad

### run-squad.py
