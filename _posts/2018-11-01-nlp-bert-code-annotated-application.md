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
    - [truncate-seq-pair(extract-feature中)](#truncate-seq-pairextract-feature%e4%b8%ad)
    - [read-examples](#read-examples)
    - [convert-examples-to-features](#convert-examples-to-features)
    - [input-fn-builder](#input-fn-builder)
    - [model-fn-builder](#model-fn-builder)
    - [main](#main)
- [pretrain](#pretrain)
  - [create-pretraining-data.py](#create-pretraining-datapy)
    - [工具函数与类](#%e5%b7%a5%e5%85%b7%e5%87%bd%e6%95%b0%e4%b8%8e%e7%b1%bb)
      - [TrainingInstance](#traininginstance)
      - [create-int-feature](#create-int-feature)
      - [create-float-feature](#create-float-feature)
      - [write-instance-to-example-files](#write-instance-to-example-files)
    - [main](#main-1)
    - [create-training-instances](#create-training-instances)
    - [truncate-seq-pair(create-pretrain中)](#truncate-seq-paircreate-pretrain%e4%b8%ad)
    - [create-instances-from-document](#create-instances-from-document)
    - [create-masked-lm-predictions](#create-masked-lm-predictions)
  - [run-pretraining.py](#run-pretrainingpy)
- [classification](#classification)
  - [run-classifier.py](#run-classifierpy)
- [squad](#squad)
  - [run-squad.py](#run-squadpy)
- [可视化分析](#%e5%8f%af%e8%a7%86%e5%8c%96%e5%88%86%e6%9e%90)

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

#### truncate-seq-pair(extract-feature中)

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
    # yield_single_examples参数是True时，会把一个batch的结果拆成batch条结果。
    # 如果是False，不分解，当结果的第一维不是batch_size时要这么用~
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

将输入文件转换成tfrecords格式

#### 工具函数与类

##### TrainingInstance

有以下几个成员变量：

+ tokens
+ segment_ids
+ is_random_next
+ masked_lm_positions
+ masked_lm_labels

```python
class TrainingInstance(object):
  """A single training instance (sentence pair)."""

  def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
               is_random_next):
    self.tokens = tokens
    self.segment_ids = segment_ids
    self.is_random_next = is_random_next
    self.masked_lm_positions = masked_lm_positions
    self.masked_lm_labels = masked_lm_labels

  def __str__(self):
    s = ""
    s += "tokens: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.tokens]))
    s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
    s += "is_random_next: %s\n" % self.is_random_next
    s += "masked_lm_positions: %s\n" % (" ".join(
        [str(x) for x in self.masked_lm_positions]))
    s += "masked_lm_labels: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.masked_lm_labels]))
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()
```

##### create-int-feature

生成int特征

```python
def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature
```

##### create-float-feature

生成float特征

```python
def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature
```

##### write-instance-to-example-files

将结果落盘

```python
def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_files):
  """Create TF example files from `TrainingInstance`s."""
  writers = []
  for output_file in output_files:
    writers.append(tf.python_io.TFRecordWriter(output_file))

  writer_index = 0

  total_written = 0
  for (inst_index, instance) in enumerate(instances):
    # 使用convert_tokens_to_ids将tokens转换为对应的input_ids
    input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = list(instance.segment_ids)
    assert len(input_ids) <= max_seq_length

    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    masked_lm_positions = list(instance.masked_lm_positions)
    masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
    masked_lm_weights = [1.0] * len(masked_lm_ids)

    while len(masked_lm_positions) < max_predictions_per_seq:
      masked_lm_positions.append(0)
      masked_lm_ids.append(0)
      masked_lm_weights.append(0.0)

    next_sentence_label = 1 if instance.is_random_next else 0

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
    features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
    features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
    features["next_sentence_labels"] = create_int_feature([next_sentence_label])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    writers[writer_index].write(tf_example.SerializeToString())
    writer_index = (writer_index + 1) % len(writers)

    total_written += 1

    if inst_index < 20:
      tf.logging.info("*** Example ***")
      tf.logging.info("tokens: %s" % " ".join(
          [tokenization.printable_text(x) for x in instance.tokens]))

      for feature_name in features.keys():
        feature = features[feature_name]
        values = []
        if feature.int64_list.value:
          values = feature.int64_list.value
        elif feature.float_list.value:
          values = feature.float_list.value
        tf.logging.info(
            "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

  for writer in writers:
    writer.close()

  tf.logging.info("Wrote %d total instances", total_written)
```

#### main

首先建立tokenizer

```python
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
```

然后create_training_instances

```python
  rng = random.Random(FLAGS.random_seed)
  instances = create_training_instances(
      input_files, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
      FLAGS.short_seq_prob, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
      rng)
```

然后把结果落盘：

```python
  write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                  FLAGS.max_predictions_per_seq, output_files)
```

#### create-training-instances

```python
def create_training_instances(input_files, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng):
  """Create `TrainingInstance`s from raw text."""
  all_documents = [[]]

  # Input file format:
  # (1) One sentence per line. These should ideally be actual sentences, not
  # entire paragraphs or arbitrary spans of text. (Because we use the
  # sentence boundaries for the "next sentence prediction" task).
  # (2) Blank lines between documents. Document boundaries are needed so
  # that the "next sentence prediction" task doesn't span between documents.
  # (1) 一行一句话，最好就是一句完整的话，而不是一段话或者半句话（因为会使用句子边界来给next sentence prediction任务用）
  # (2) 文档间用空行隔开
  for input_file in input_files:
    with tf.gfile.GFile(input_file, "r") as reader:
      while True:
        line = tokenization.convert_to_unicode(reader.readline())
        if not line:
          break
        line = line.strip()

        # Empty lines are used as document delimiters
        # 空行被视为文档的分隔符
        if not line:
          all_documents.append([])
        tokens = tokenizer.tokenize(line)
        if tokens:
          all_documents[-1].append(tokens)

  # Remove empty documents
  all_documents = [x for x in all_documents if x]
  # 全部load到内存，所以如果有非常多的训练语料，建议在外面拆分成多个小文件，然后多进程调用这个函数。
  rng.shuffle(all_documents)

  vocab_words = list(tokenizer.vocab.keys())
  instances = []
  for _ in range(dupe_factor):
    for document_index in range(len(all_documents)):
      instances.extend(
          ## 调用这个函数生成一条ins
          create_instances_from_document(
              all_documents, document_index, max_seq_length, short_seq_prob,
              masked_lm_prob, max_predictions_per_seq, vocab_words, rng))

  rng.shuffle(instances)
  return instances
```

#### truncate-seq-pair(create-pretrain中)

```python
def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
  """Truncates a pair of sequences to a maximum sequence length."""
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_num_tokens:
      break
    ## a和b两个部分，每次对比较长的那个进行trunc，当两部分差不多长时，交替trunc，保证不要一直对一个部分去trunc。。
    trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
    assert len(trunc_tokens) >= 1

    # We want to sometimes truncate from the front and sometimes from the
    # back to add more randomness and avoid biases.
    # 一半的概率扔掉头，一半概率扔掉尾
    if rng.random() < 0.5:
      del trunc_tokens[0]
    else:
      trunc_tokens.pop()
```

#### create-instances-from-document

```python
def create_instances_from_document(
    all_documents, document_index, max_seq_length, short_seq_prob,
    masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
  """Creates `TrainingInstance`s for a single document."""
  # 获取当前文档
  document = all_documents[document_index]

  # Account for [CLS], [SEP], [SEP]
  max_num_tokens = max_seq_length - 3

  # We *usually* want to fill up the entire sequence since we are padding
  # to `max_seq_length` anyways, so short sequences are generally wasted
  # computation. However, we *sometimes*
  # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
  # sequences to minimize the mismatch between pre-training and fine-tuning.
  # The `target_seq_length` is just a rough target however, whereas
  # `max_seq_length` is a hard limit.
  # 有一定的比例，如10%的概率，我们使用比较短的序列长度，以缓解预训练的长序列和finetune阶段（可能的）短序列的不一致情况
  target_seq_length = max_num_tokens
  if rng.random() < short_seq_prob:
    target_seq_length = rng.randint(2, max_num_tokens)

  # We DON'T just concatenate all of the tokens from a document into a long
  # sequence and choose an arbitrary split point because this would make the
  # next sentence prediction task too easy. Instead, we split the input into
  # segments "A" and "B" based on the actual "sentences" provided by the user
  # input.
  # A和B都使用完整的句子，而非半句话
  instances = []
  current_chunk = []
  current_length = 0
  i = 0
  while i < len(document):
    segment = document[i] #获取当前的一句话，并扔到current_chunk里
    current_chunk.append(segment)
    current_length += len(segment) # curren_length是这些句子的总长度
    ## 要么遍历完文档了，要么句子总长度已经超过target_seq_length了（够一条样本了）
    if i == len(document) - 1 or current_length >= target_seq_length:
      if current_chunk:
        # `a_end` is how many segments from `current_chunk` go into the `A`
        # (first) sentence.
        # a_end是当前chunk的多少句话可以放到A的候选中
        a_end = 1
        if len(current_chunk) >= 2:
          a_end = rng.randint(1, len(current_chunk) - 1)

        tokens_a = []
        # 从第0句到第a_end句，首尾相连！！扔到tokens_a里
        for j in range(a_end):
          tokens_a.extend(current_chunk[j])

        tokens_b = []
        # Random next
        is_random_next = False
        # 当文档只有一句话，或者以0.5的概率，从其他文档随机采样
        if len(current_chunk) == 1 or rng.random() < 0.5:
          is_random_next = True
          target_b_length = target_seq_length - len(tokens_a)

          # This should rarely go for more than one iteration for large
          # corpora. However, just to be careful, we try to make sure that
          # the random document is not the same as the document
          # we're processing.
          for _ in range(10):
            random_document_index = rng.randint(0, len(all_documents) - 1)
            if random_document_index != document_index:
              break
          # 随机一个文档出来
          random_document = all_documents[random_document_index]
          random_start = rng.randint(0, len(random_document) - 1)
          # 在这篇随机出的文档中，随机出一个位置，这个位置开始的后面所有句子，首尾相连地连起来！！作为tokens_b
          for j in range(random_start, len(random_document)):
            tokens_b.extend(random_document[j])
            if len(tokens_b) >= target_b_length:
              break
          # We didn't actually use these segments so we "put them back" so
          # they don't go to waste.
          num_unused_segments = len(current_chunk) - a_end
          i -= num_unused_segments
        # Actual next
        else:
          is_random_next = False
          for j in range(a_end, len(current_chunk)):
            tokens_b.extend(current_chunk[j])
        truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

        assert len(tokens_a) >= 1
        assert len(tokens_b) >= 1

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
          tokens.append(token)
          segment_ids.append(0)

        tokens.append("[SEP]")
        segment_ids.append(0)

        for token in tokens_b:
          tokens.append(token)
          segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        (tokens, masked_lm_positions,
         masked_lm_labels) = create_masked_lm_predictions(
             tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
        instance = TrainingInstance(
            tokens=tokens,
            segment_ids=segment_ids,
            is_random_next=is_random_next,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        instances.append(instance)
      current_chunk = []
      current_length = 0
    i += 1

  return instances
```

#### create-masked-lm-predictions

首先定义了一个namedtuple：

```python
MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])
```

然后是这个函数的实现：

```python
def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
  """Creates the predictions for the masked LM objective."""

  cand_indexes = []
  for (i, token) in enumerate(tokens):
    if token == "[CLS]" or token == "[SEP]":
      continue
    # wordpieces涉及到Whole Word Masking(WWM)。wordpieces中，同一个词的第一个token没有marker，中间的token以##开头，所以见到##开头的话，就和前一个词的下标拼到一起去！！
    # 如果是中文，可能参考albert_zh的实现：https://github.com/daiwk/dl-frame/blob/43d120ad5cb6e1766b0a0b26f8f5c5376ec36069/demos/thirdparty/albert_zh/resources/create_pretraining_data_roberta.py#L261
    # Whole Word Masking means that if we mask all of the wordpieces
    # corresponding to an original word. When a word has been split into
    # WordPieces, the first token does not have any marker and any subsequence
    # tokens are prefixed with ##. So whenever we see the ## token, we
    # append it to the previous set of word indexes.
    #
    # Note that Whole Word Masking does *not* change the training code
    # at all -- we still predict each WordPiece independently, softmaxed
    # over the entire vocabulary.
    if (FLAGS.do_whole_word_mask and len(cand_indexes) >= 1 and
        token.startswith("##")):
      cand_indexes[-1].append(i)
    else:
      cand_indexes.append([i])

  rng.shuffle(cand_indexes)

  output_tokens = list(tokens)

  num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(tokens) * masked_lm_prob))))

  masked_lms = []
  covered_indexes = set()
  for index_set in cand_indexes:
    if len(masked_lms) >= num_to_predict:
      break
    # If adding a whole-word mask would exceed the maximum number of
    # predictions, then just skip this candidate.
    if len(masked_lms) + len(index_set) > num_to_predict:
      continue
    is_any_index_covered = False
    for index in index_set:
      if index in covered_indexes:
        is_any_index_covered = True
        break
    if is_any_index_covered:
      continue
    for index in index_set:
      covered_indexes.add(index)

      masked_token = None
      # 80% of the time, replace with [MASK]
      if rng.random() < 0.8:
        masked_token = "[MASK]"
      else:
        # 10% of the time, keep original
        if rng.random() < 0.5:
          masked_token = tokens[index]
        # 10% of the time, replace with random word
        else:
          masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

      output_tokens[index] = masked_token

      masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
  assert len(masked_lms) <= num_to_predict
  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)

  return (output_tokens, masked_lm_positions, masked_lm_labels)
```

### run-pretraining.py

读入tfrecords格式的训练样本，进行训练


## classification

使用自己的数据集基于现有模型进行finetune

### run-classifier.py

## squad

### run-squad.py

## 可视化分析

参考[https://towardsdatascience.com/deconstructing-bert-part-2-visualizing-the-inner-workings-of-attention-60a16d86b5c1](https://towardsdatascience.com/deconstructing-bert-part-2-visualizing-the-inner-workings-of-attention-60a16d86b5c1)

github：[https://github.com/jessevig/bertviz](https://github.com/jessevig/bertviz)
