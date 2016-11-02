---
layout: post
category: "nlp"
title: "lstm crf"
tags: [nlp, natural language processing, lstm crf, lstm, crf]
---

# **1. 【2015】Bidirectional LSTM-CRF Models for Sequence Tagging**

[论文地址](../assets/Bidirectional LSTM-CRF Models for Sequence Tagging.pdf)

# **2. 【2016】Conditional Random Fields as Recurrent Neural Networks**

[论文地址](../assets/Conditional Random Fields as Recurrent Neural Networks.pdf)

# **3. 【2016,ACL】Deep Recurrent Models with Fast-Forward Connections for Neural Machine Translation**
[论文地址](../assets/Deep Recurrent Models with Fast-Forward Connections for Neural Machine Translation.pdf)

# **4. 在paddlepaddle上实现**

掉渣天的徐老师已经在开源版本的paddlepaddle实现了[rnn+crf](https://github.com/baidu/Paddle/tree/develop/demo/sequence_tagging)，我们直接拿来学习学习就好啦！！！

md。。内部版本和开源版本不一致。。我们需要把开源版本重新安装一遍……

先看一下数据集的说明：
[http://www.clips.uantwerpen.be/conll2000/chunking/](http://www.clips.uantwerpen.be/conll2000/chunking/)

```shell
#[NP He ] [VP reckons ] [NP the current account deficit ] [VP will narrow ] [PP to ] [NP only # 1.8 billion ] [PP in ] [NP September ] .
#
#He        PRP  B-NP
#reckons   VBZ  B-VP
#the       DT   B-NP
#current   JJ   I-NP
#account   NN   I-NP
#deficit   NN   I-NP
#will      MD   B-VP
#narrow    VB   I-VP
#to        TO   B-PP
#only      RB   B-NP
##         #    I-NP
#1.8       CD   I-NP
#billion   CD   I-NP
#in        IN   B-PP
#September NNP  B-NP
#.         .    O
#
#The O chunk tag is used for tokens which are not part of any chunk
#
#
```

首先，需要将文件组织成三列，第一列是单词（中文的单字），第二列是tag_pos（part-of-speech tag as derived by the Brill tagger），第三列是标签tag（B-x表示开始，I-x表示中间）。比如，我们写好了test.txt，那么我们运行**gzip test.txt**，这样，就生成了test.txt.gz（解压的时候gzip -dc test.txt.gz > test.txt，就会生成test.txt[**如果不用-c参数，会默认删掉test.txt.gz**]）。

然后我们来仔细看一下dataprovider：

```python

# Feature combination patterns.(crf要用到的特征？)
# [[-1,0], [0,0]]  means previous token at column 0 and current token at 
# column 0 are combined as one feature.
patterns = [
    [[-2,0]],
    [[-1,0]],
    [[0,0]],
    [[1,0]],
    [[2,0]],

    [[-1,0], [0,0]], 
    [[0,0], [1,0]], 

    [[-2,1]],
    [[-1,1]],
    [[0,1]],
    [[1,1]],
    [[2,1]],
    [[-2,1], [-1,1]],
    [[-1,1], [0,1]],
    [[0,1], [1,1]],
    [[1,1], [2,1]],

    [[-2,1], [-1,1], [0,1]],
    [[-1,1], [0,1], [1,1]],
    [[0,1], [1,1], [2,1]],
]

dict_label = {
 'B-ADJP': 0,
 'I-ADJP': 1,
 'B-ADVP': 2,
 'I-ADVP': 3,
 'B-CONJP': 4,
 'I-CONJP': 5,
 'B-INTJ': 6,
 'I-INTJ': 7,
 'B-LST': 8,
 'I-LST': 9,
 'B-NP': 10,
 'I-NP': 11,
 'B-PP': 12,
 'I-PP': 13,
 'B-PRT': 14,
 'I-PRT': 15,
 'B-SBAR': 16,
 'I-SBAR': 17,
 'B-UCP': 18,
 'I-UCP': 19,
 'B-VP': 20,
 'I-VP': 21,
 'O': 22
}

def make_features(sequence):
    length = len(sequence)
    num_features = len(sequence[0])
    def get_features(pos):
        if pos < 0:#比如，特征是[[-1,0]]，但现在是第0个字，传进来pos就是0-1，但不能0-1啊，就标为#B1（向前超出了1个）
            return ['#B%s' % -pos] * num_features
        if pos >= length:#同样道理，向后超出了多少个
            return ['#E%s' % (pos - length + 1)] * num_features
        return sequence[pos]

    for i in xrange(length):
        for pattern in patterns:
            fname = '/'.join([get_features(i+pos)[f] for pos, f in pattern]) #例如，feature是[[0,1], [1,1], [2,1]],那么，fname可能是NNP/NNP/POS，如果是[[2,1]],那么，fname就是POS
            sequence[i].append(fname)


create_dictionaries函数中：

#cutoff: a list of numbers. If count of a feature is smaller than this,              
# it will be ignored.          
cutoff = [3, 1, 0]
cutoff += [3] * len(patterns)

cutoff=[3, 1, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]                                                      

#if oov_policy[i] is OOV_POLICY_USE, id 0 is reserved for OOV features of            
#i-th column.                                                                        

OOV_POLICY_IGNORE = 0
OOV_POLICY_USE = 1
OOV_POLICY_ERROR = 2

oov_policy = [OOV_POLICY_IGNORE, OOV_POLICY_ERROR, OOV_POLICY_ERROR]
oov_policy += [OOV_POLICY_IGNORE] * len(patterns)

oov_policy=[0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

                                                                             
#return a list of dict for each column
def create_dictionaries(filename, cutoff, oov_policy):
    def add_to_dict(sequence, dicts):
        num_features = len(dicts)
        for features in sequence:
            l = len(features)
            assert l == num_features, "Wrong number of features " + line
            for i in xrange(l):
                if features[i] in dicts[i]:
                    dicts[i][features[i]] += 1
                else:
                    dicts[i][features[i]] = 1

    num_features = len(cutoff)
    dicts = []
    for i in xrange(num_features):
        dicts.append(dict())

    f = gzip.open(filename, 'rb')

    sequence = []

    for line in f:
        line = line.strip()
        if not line:#空串，用于区分不同句子
            make_features(sequence)
            add_to_dict(sequence, dicts)
            sequence = []
            continue
        features = line.split(' ')
        sequence.append(features)


    for i in xrange(num_features):
        dct = dicts[i]
        n = 1 if oov_policy[i] == OOV_POLICY_USE else 0
        todo = []
        for k, v in dct.iteritems():
            if v < cutoff[i]:
                todo.append(k)
            else:
                dct[k] = n
                n += 1
            
        if oov_policy[i] == OOV_POLICY_USE:
            # placeholder so that len(dct) will be the number of features
            # including OOV
            dct['#OOV#'] = 0

        logger.info('column %d dict size=%d, ignored %d' % (i, n, len(todo)))
        for k in todo:
            del dct[k]

    f.close()
    return dicts

```

initializer的流程：

create_dictionaries的流程：

+ 初始化dicts数组，数组长度为num_features(即len(cutoff))，每个元素是一个空字典
+ 初始化sequence = []
+ 遍历文件，
    + features = line.split(' ')，将features放进sequence中。（也就是说sequence最开始会有3个元素，词，tag_pos，tag）
    + 每读到空行，表示一个序列结束了，
        + 进行make_features，将这一个序列（比如训练数据有100个序列，这是第33个，而这个序列有三个字符）sequence[33]变成['jetliners', 'NNS', 'I-NP', "'s", '747', 'jetliners', '.', '#E1', '747/jetliners', 'jetliners/.']之类的，长度为feature+3个数的数组（其实就是最开始放进去的那3个元素，再加上每个feature拿到对应的值）
        + 进行add_to_dict，
        + 重置sequence = []，并continue
    + aaa
+ 啦啦啦
 