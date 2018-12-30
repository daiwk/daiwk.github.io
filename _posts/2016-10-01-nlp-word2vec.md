---
layout: post
category: "nlp"
title: "word2vec"
tags: [word2vec, ngram, nnlm, cbow, c-skip-gram, 统计语言模型]
---

目录

<!-- TOC -->

- [1. 统计语言模型](#1-统计语言模型)
    - [N-gram模型](#n-gram模型)
    - [神经网络语言模型（NNLM）](#神经网络语言模型nnlm)
- [2. CBOW(Continuous Bag-of-Words)](#2-cbowcontinuous-bag-of-words)
- [3. Continuous skip-gram](#3-continuous-skip-gram)
- [4. NCE](#4-nce)
- [5. 面试常见问题](#5-面试常见问题)
- [x. tensorflow的简单实现](#x-tensorflow的简单实现)
    - [简介](#简介)
    - [代码解读](#代码解读)
        - [读数据](#读数据)
        - [建立数据集](#建立数据集)
        - [生成skipgram的一个batch](#生成skipgram的一个batch)
        - [定义模型](#定义模型)
            - [placeholder等](#placeholder等)
            - [网络参数](#网络参数)
            - [loss](#loss)
            - [计算cos](#计算cos)
            - [其他](#其他)
        - [训练](#训练)
        - [可视化](#可视化)
- [y1. tensorflow的高级实现1](#y1-tensorflow的高级实现1)
- [y2. tensorflow的高级实现2](#y2-tensorflow的高级实现2)

<!-- /TOC -->

参考cdsn博客：[http://blog.csdn.net/zhaoxinfan/article/details/27352659](http://blog.csdn.net/zhaoxinfan/article/details/27352659)

参考paddlepaddle book: [https://github.com/PaddlePaddle/book/blob/develop/04.word2vec/README.cn.md](https://github.com/PaddlePaddle/book/blob/develop/04.word2vec/README.cn.md)

参考fasttext及更多cbow/skip-gram：[https://daiwk.github.io/posts/nlp-fasttext.html](https://daiwk.github.io/posts/nlp-fasttext.html)

Word2vec的原理主要涉及到**统计语言模型**（包括N-gram模型和神经网络语言模型(nnlm)），**continuous bag-of-words**模型以及**continuous skip-gram**模型。

语言模型旨在为语句的联合概率函数`\(P(w_1,...,w_T)\)`建模。语言模型的目标是，希望模型对有意义的句子赋予大概率，对没意义的句子赋予小概率。 常用条件概率表示语言模型：

`\[
P(w_1, ..., w_T) = \prod_{t=1}^TP(w_t | w_1, ... , w_{t-1})
\]`

# 1. 统计语言模型
## N-gram模型

n-gram是一种重要的文本表示方法，表示一个文本中连续的n个项。基于具体的应用场景，每一项可以是一个字母、单词或者音节。一般用每个n-gram的历史n-1个词语组成的内容来预测第n个词。

## 神经网络语言模型（NNLM）

Yoshua Bengio等科学家就于2003年在著名论文 Neural Probabilistic Language Models中介绍如何学习一个神经元网络表示的词向量模型。神经概率语言模型（Neural Network Language Model，NNLM）通过**一个线性映射和一个非线性隐层连接，**同时学习了语言模型和词向量，即通过学习大量语料得到词语的向量表达，通过这些向量得到整个句子的概率。用这种方法学习语言模型可以**克服维度灾难（curse of dimensionality）,即训练和测试数据不同导致的模型不准。**

实际上越远的词语其实对该词的影响越小，那么如果考虑一个n-gram, 每个词都只受其前面n-1个词的影响，则有：

`\[
P(w_1, ..., w_T) = \prod_{t=n}^TP(w_t|w_{t-1}, w_{t-2}, ..., w_{t-n+1})
\]`

给定一些真实语料，这些语料中都是有意义的句子，N-gram模型的优化目标则是最大化目标函数:

`\[
\frac{1}{T}\sum_t f(w_t, w_{t-1}, ..., w_{t-n+1};\theta) + R(\theta)
\]`

其中，`\(f(w_t, w_{t-1}, ..., w_{t-n+1})\)`表示根据历史n-1个词得到当前词`\(w_t\)`的条件概率，`\(R(\theta)\)`表示参数正则项。

<html>
<br/>
<img src='../assets/word2vec-nnlm.png' style='max-width: 400px'/>
<img src='../assets/word2vec-nnlm-noted.jpg' style='max-width: 400px'/>
<br/>
</html>

+ 对于每个样本，模型输入`\(w_{t-n+1},...w_{t-1}\)`，输出句子第t个词为字典中`\(|V|\)`个词的概率。每个输入词`\(w_{t-n+1},...w_{t-1}\)`通过矩阵`\(C\)`映射到词向量`\(C(w_{t-n+1}),...C(w_{t-1})\)`

+ 然后所有词语的词向量连接成一个大向量，并经过一个非线性映射得到历史词语的隐层表示：

`\[
g=Utanh(\theta^Tx + b_1) + Wx + b_2
\]`

其中，x为所有词语的词向量连接成的大向量，表示文本历史特征；`\(g\)`表示未经归一化的所有输出单词概率，`\(g_i\)`表示未经归一化的字典中第i个单词的输出概率。其他参数为词向量层到隐层连接的参数。

+ 根据softmax的定义，通过归一化`\(g_i\)`, 生成目标词`\(w_t\)`的概率为：

`\[
P(w_t | w_1, ..., w_{t-n+1}) = \frac{e^{g_{w_t}}}{\sum_i^{|V|} e^{g_i}}
\]`

+ 整个网络的损失值(cost)为多类分类交叉熵，用公式表示为：

`\[
J(\theta) = -\sum_{i=1}^N\sum_{c=1}^{|V|}y_k^{i}log(softmax(g_k^i))
\]`

其中`\(y_k^i\)`表示第i个样本第k类的真实标签(0或1)，`\(softmax(g_k^i)\)`表示第i个样本第k类softmax输出的概率。

# 2. CBOW(Continuous Bag-of-Words)

CBOW模型通过一个词的上下文（各N个词）预测当前词。当N=2时，模型如下图所示：

<html>
<br/>

<img src='../assets/word2vec-cbow.png' style='max-height: 300px'/>
<br/>

</html>

具体来说，不考虑上下文的词语输入顺序，CBOW是用上下文词语的词向量的均值来预测当前词。即：

`\[
context = \frac{x_{t-1} + x_{t-2} + x_{t+1} + x_{t+2}}{4}
\]`

其中`\(x_t\)`为第t个词的词向量，分类分数（score）向量 `\(z=U*context\)`，最终的分类y采用softmax，损失函数采用多类分类交叉熵。

CBOW经常结合hierarchical softmax一起实现，详见[https://daiwk.github.io/posts/nlp-fasttext.html#02-%E5%88%86%E5%B1%82softmax](https://daiwk.github.io/posts/nlp-fasttext.html#02-%E5%88%86%E5%B1%82softmax)

# 3. Continuous skip-gram

CBOW的好处是对上下文词语的分布在词向量上进行了平滑，去掉了噪声，因此在小数据集上很有效。而Skip-gram的方法中，**用一个词预测其上下文，得到了当前词上下文的很多样本，因此可用于更大的数据集。**

<html>
<br/>
<img src='../assets/word2vec-skipgram.png' style='max-height: 300px'/>
<br/>

</html>

如上图所示，Skip-gram模型的具体做法是，将一个词的词向量映射到2n个词的词向量（2n表示当前输入词的前后各n个词），然后分别通过softmax得到这2n个词的分类损失值之和。

参考[https://blog.csdn.net/u014595019/article/details/54093161](https://blog.csdn.net/u014595019/article/details/54093161)

『我当时使用的是Hierarchical Softmax+CBOW的模型。给我的感觉是比较累，既要费力去写huffman树，还要自己写计算梯度的代码，完了按层softmax速度还慢。这次我决定用tensorflow来写，除了极大的精简了代码以外，可以使用gpu对运算进行加速。此外，这次使用了**负采样(negative sampling)+skip-gram**模型，从而**避免了使用Huffman树导致训练速度变慢**的情况，**适合大规模的文本**。』

而且，在tf中的实现```tensorflow/tensorflow/examples/tutorials/word2vec/word2vec_basic.py```，也是基于skip-gram+nce_loss的。

# 4. NCE

参考[https://blog.csdn.net/itplus/article/details/37998797](https://blog.csdn.net/itplus/article/details/37998797)

# 5. 面试常见问题

参考 [https://blog.csdn.net/zhangxb35/article/details/74716245](https://blog.csdn.net/zhangxb35/article/details/74716245)

# x. tensorflow的简单实现

讲解：[https://www.tensorflow.org/tutorials/word2vec](https://www.tensorflow.org/tutorials/word2vec)

代码：[https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/examples/tutorials/word2vec/word2vec_basic.py](https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/examples/tutorials/word2vec/word2vec_basic.py)

## 简介

使用maximum likelihood principle，最大化给定previous words `\(h\)`，下一个词`\(w_t\)`的概率（使用softmax定义）：

`\[
\begin{align}
P(w_t | h) &= \text{softmax} (\text{score} (w_t, h)) \\
           &= \frac{\exp \{ \text{score} (w_t, h) \} }
             {\sum_\text{Word w' in Vocab} \exp \{ \text{score} (w', h) \} }
\end{align}
\]`

需要最大化的对应的log-likelihood就是：

`\[
\begin{align}
 J_\text{ML} &= \log P(w_t | h) \\
  &= \text{score} (w_t, h) -
     \log \left( \sum_\text{Word w' in Vocab} \exp \{ \text{score} (w', h) \} \right).
\end{align}
\]`

如果直接硬算，对于每个时间步，都需要遍历词典大小的空间，显然效率是不行的。在word2vec中，将这样一个多分类问题，变成了一个区分目标词`\(w_t\)`和k个noise words`\(\tilde w\)`的二分类问题。下图是cbow的示例，skipgram正好是倒过来。

<html>
<br/>
<img src='../assets/nce-nplm.png' style='max-width: 400px'/>
<br/>
</html>

数学上，目标就是要最大化：

`\[
J_\text{NEG} = \log Q_\theta(D=1 |w_t, h) +
  k \mathop{\mathbb{E}}_{\tilde w \sim P_\text{noise}}
     \left[ \log Q_\theta(D = 0 |\tilde w, h) \right]
\]`

其中，`\(Q_\theta(D=1 | w, h)\)`是使用学到的embedding vector `\(\theta\)`，在给定数据集`\(D\)`中的上下文`\(h\)`，看到词`\(w\)`的概率。在实践中，我们通过从噪声分布中抽取`\(k\)`个对比词来逼近期望值（即计算[蒙特卡洛平均值](https://en.wikipedia.org/wiki/Monte_Carlo_integration)）。

直观地理解，这个目标就是希望预测为`\(w_t\)`的概率尽可能大，同时预测为非`\(\tilde w\)`的概率尽可能大，也就是，**希望预测为真实词的概率尽量大，预测为noise word的概率尽量小**。在极限情况下，这可以近似为softmax，但这计算量比softmax小很多。这就是所谓的[negative sampling](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)。

tensorflow有一个很类似的损失函数[noise-contrastive estimation(NCE)](https://papers.nips.cc/paper/5165-learning-word-embeddings-efficiently-with-noise-contrastive-estimation.pdf)，即『噪声对比估算』，```tf.nn.nce_loss()```。

针对句子

```
the quick brown fox jumped over the lazy dog
```

如果使用window_size=1（一个词**左右各**采一个词），那么对于CBOW，就有：

```
([the, brown], quick), ([quick, fox], brown), ([brown, jumped], fox), ...
```

而对于skipgram，则有：

```
(quick, the), (quick, brown), (brown, quick), (brown, fox), ...
```

在训练步`\(t\)`中，例如要用```quick```来预测```the```，我们从某个噪声分布（通常为一元分布`\(P(w)\)`）中抽取```num_noise```个噪声对比样本。假设num_noise=1，并且将```sheep```选为噪声样本。那么在`\(t\)`时刻的目标是：

`\[
J^{(t)}_\text{NEG} = \log Q_\theta(D=1 | \text{the, quick}) +
  \log(Q_\theta(D=0 | \text{sheep, quick}))
\]`



画图时，可以使用[t-SNE](https://lvdmaaten.github.io/tsne/)的降维方法，将高维向量映射到2维空间。

<html>
<br/>
<img src='../assets/linear-relationships.png' style='max-width: 400px'/>
<br/>
</html>

## 代码解读

### 读数据

```python
# Read the data into a list of strings.
def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words."""
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data
```

调用：

```python
vocabulary = read_data(filename)
```

### 建立数据集

build一个dataset：

```python
def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1)) ## 取出词频top n_words-1的词，词频高的index小
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    index = dictionary.get(word, 0)
    if index == 0:  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary
```

调用：

```python
data, count, dictionary, reverse_dictionary = build_dataset(
    vocabulary, vocabulary_size)
```

### 生成skipgram的一个batch

生成一个batch的方法如下：

```python
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
  if data_index + span > len(data):
    data_index = 0
  buffer.extend(data[data_index:data_index + span])
  data_index += span
  for i in range(batch_size // num_skips):
    context_words = [w for w in range(span) if w != skip_window]
    words_to_use = random.sample(context_words, num_skips)
    for j, context_word in enumerate(words_to_use):
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[context_word]
    if data_index == len(data):
      buffer.extend(data[0:span])
      data_index = span
    else:
      buffer.append(data[data_index])
      data_index += 1
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels
```

调用：

```python
batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
```

### 定义模型

#### placeholder等

先定义一下validation set

```python
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
```

然后定义一下input和label的placeholder：

```python
graph = tf.Graph()

with graph.as_default():

  # Input data.
  with tf.name_scope('inputs'):
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
```

#### 网络参数

定义：

+ vocabulary_size x embedding_size的embedding
+ vocabulary_size x embedding_size的nce_weights
+ vocabulary_size的nce_biases

```python
  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    with tf.name_scope('embeddings'):
      embeddings = tf.Variable(
          tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
      embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    with tf.name_scope('weights'):
      nce_weights = tf.Variable(
          tf.truncated_normal(
              [vocabulary_size, embedding_size],
              stddev=1.0 / math.sqrt(embedding_size)))
    with tf.name_scope('biases'):
      nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
```

#### loss

在NCE的实现中，使用的是log_uniform_candidate_sampler：

+ 会在[0, range_max)中采样出一个整数k(**k相当于词的id**)
+ P(k) = (log(k + 2) - log(k + 1)) / log(range_max + 1)

`\[
\begin{aligned}
P(k)&=\frac{1}{log(range\_max+1)}log(\frac{k+2}{k+1}) \\ 
 &= \frac{1}{log(range\_max+1)}log(1+\frac{1}{k+1}) \\
\end{aligned}
\]`

**k越大，被采样到的概率越小。**而我们的词典中，可以发现**词频高**的**index小**，所以高词频的词会被**优先**采样为负样本。

下面定义loss

```python
  with tf.name_scope('loss'):
    loss = tf.reduce_mean(
        tf.nn.nce_loss(
            weights=nce_weights,
            biases=nce_biases,
            labels=train_labels,
            inputs=embed,
            num_sampled=num_sampled,
            num_classes=vocabulary_size))
  # Construct the SGD optimizer using a learning rate of 1.0.
  with tf.name_scope('optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
```

#### 计算cos

只是为了看看迭代过程中，validtion set里每个词最近的topk词，看看效果。

```python
  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  ## 使用valid_dataset做输入，normalized_embeddings做参数，进行lookup得到的emb
  valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                            valid_dataset)
  # (valid_size, emb_size) x (emb_size, vocab_size) = (valid_size, vocab_size)
  # 因为valid_emb和norm_emb都是emb/norm(emb)，所以两者的乘积就是cos
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)
```

#### 其他

```python
  # Merge all summaries.
  merged = tf.summary.merge_all()

  # Add variable initializer.
  init = tf.global_variables_initializer()

  # Create a saver.
  saver = tf.train.Saver()
```

### 训练

```python
with tf.Session(graph=graph) as session:
  # Open a writer to write summaries.
  writer = tf.summary.FileWriter(FLAGS.log_dir, session.graph)

  # We must initialize all variables before we use them.
  init.run()
  print('Initialized')

  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(batch_size, num_skips,
                                                skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    # Define metadata variable.
    run_metadata = tf.RunMetadata()

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    # Also, evaluate the merged op to get all summaries from the returned "summary" variable.
    # Feed metadata variable to session for visualizing the graph in TensorBoard.
    _, summary, loss_val = session.run(
        [optimizer, merged, loss],
        feed_dict=feed_dict,
        run_metadata=run_metadata)
    average_loss += loss_val

    # Add returned summaries to writer in each step.
    writer.add_summary(summary, step)
    # Add metadata to visualize the graph for the last run.
    if step == (num_steps - 1):
      writer.add_run_metadata(run_metadata, 'step%d' % step)

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step ', step, ': ', average_loss)
      average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = '%s %s,' % (log_str, close_word)
        print(log_str)
  final_embeddings = normalized_embeddings.eval()

  # Write corresponding labels for the embeddings.
  with open(FLAGS.log_dir + '/metadata.tsv', 'w') as f:
    for i in xrange(vocabulary_size):
      f.write(reverse_dictionary[i] + '\n')

  # Save the model for checkpoints.
  saver.save(session, os.path.join(FLAGS.log_dir, 'model.ckpt'))

  # Create a configuration for visualizing embeddings with the labels in TensorBoard.
  config = projector.ProjectorConfig()
  embedding_conf = config.embeddings.add()
  embedding_conf.tensor_name = embeddings.name
  embedding_conf.metadata_path = os.path.join(FLAGS.log_dir, 'metadata.tsv')
  projector.visualize_embeddings(writer, config)

writer.close()
```

### 可视化

```python
# Function to draw visualization of distance between embeddings.
def plot_with_labels(low_dim_embs, labels, filename):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(
        label,
        xy=(x, y),
        xytext=(5, 2),
        textcoords='offset points',
        ha='right',
        va='bottom')

  plt.savefig(filename)


try:
  # pylint: disable=g-import-not-at-top
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(
      perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels, os.path.join(gettempdir(), 'tsne.png'))

except ImportError as ex:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')
  print(ex)
```

# y1. tensorflow的高级实现1

如果您发现模型在**输入数据方面**存在严重**瓶颈**，您可能需要针对您的问题实现**自定义数据读取器**

[https://github.com/tensorflow/models/blob/master/tutorials/embedding/word2vec.py](https://github.com/tensorflow/models/blob/master/tutorials/embedding/word2vec.py)


# y2. tensorflow的高级实现2

如果您的模型不再受 I/O 限制，但您仍希望提高性能，则可以通过**编写自己的 TensorFlow 操作**（如[添加新操作](https://www.tensorflow.org/guide/extend/op)中所述）进一步采取措施：

[https://github.com/tensorflow/models/blob/master/tutorials/embedding/word2vec_optimized.py](https://github.com/tensorflow/models/blob/master/tutorials/embedding/word2vec_optimized.py)

