---
layout: post
category: "rl"
title: "深入浅出强化学习-chap9 基于确定性策略搜索的强化学习方法"
tags: [深入浅出强化学习, DPG, DDPG, AC, A3C]
---

目录

<!-- TOC -->

- [1. 概述](#1-%e6%a6%82%e8%bf%b0)
- [2. 随机策略与确定性策略](#2-%e9%9a%8f%e6%9c%ba%e7%ad%96%e7%95%a5%e4%b8%8e%e7%a1%ae%e5%ae%9a%e6%80%a7%e7%ad%96%e7%95%a5)
  - [2.1 随机策略](#21-%e9%9a%8f%e6%9c%ba%e7%ad%96%e7%95%a5)
  - [2.2 确定性策略](#22-%e7%a1%ae%e5%ae%9a%e6%80%a7%e7%ad%96%e7%95%a5)
  - [2.3 对比](#23-%e5%af%b9%e6%af%94)
- [3. AC框架](#3-ac%e6%a1%86%e6%9e%b6)
  - [3.1 随机策略AC方法](#31-%e9%9a%8f%e6%9c%ba%e7%ad%96%e7%95%a5ac%e6%96%b9%e6%b3%95)
  - [3.2 确定性策略AC方法（DPG）](#32-%e7%a1%ae%e5%ae%9a%e6%80%a7%e7%ad%96%e7%95%a5ac%e6%96%b9%e6%b3%95dpg)
  - [3.3 深度确定性策略梯度方法（DDPG）](#33-%e6%b7%b1%e5%ba%a6%e7%a1%ae%e5%ae%9a%e6%80%a7%e7%ad%96%e7%95%a5%e6%a2%af%e5%ba%a6%e6%96%b9%e6%b3%95ddpg)
  - [3.3 A3C(asynchronous advantage actor-critic)](#33-a3casynchronous-advantage-actor-critic)

<!-- /TOC -->


参考**《深入浅出强化学习》**

## 1. 概述

model-free的策略搜索方法可以分为随机策略搜索方法和确定性策略搜索方法。

+ 2014年以前，学者们都在发展随机策略搜索方法。因为大家认为确定性策略梯度是不存在的。
+ 2014年Silver在论文[Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf)中提出了**确定性策略理论**，即DPG。
+ 2015年DeepMind又将DPG与DQN的成功经验相结合，提出了[Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971)，即**DDPG**
+ ICML2016，提出了[Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)，即A3C（asynchronous advantage actor-critic）算法。

## 2. 随机策略与确定性策略

### 2.1 随机策略

随机策略公式为：

`\[
\pi_{\theta}(a|s)=P[a|s;\theta]
\]`

含义为，在状态`\(s\)`时，动作符合参数为`\(\theta\)`的概率分布，例如常用的高斯策略：

`\[
\pi_{\theta}(a|s)=\frac{1}{\sqrt{2\pi \sigma}}exp(-\frac{(a-f_{\theta}(s))}{2\sigma ^2})
\]`

在状态`\(s\)`处，采取的动作服从均值为`\(f_{\theta}(s)\)`，方差为`\(\sigma ^2\)`的正态分布。所以，即使在相同的状态下，每次采取的动作也可能不一样。

### 2.2 确定性策略

确定性策略的公式如下：

`\[
a=\mu_{\theta}(s)
\]`

相同的策略（即相同`\(\theta\)`），在状态`\(s\)`时，动作是唯一确定的。

### 2.3 对比

+ 确定性策略的优点在于：**需要采样的数据少**，算法效率高

随机策略的梯度计算公式：

`\[
\triangledown _{\theta}J(\pi _{\theta})=E_{s\sim \rho ^{\pi},a\sim \pi_{\theta}}[\triangledown _{\theta}log\pi_{\theta}(a|s)Q^{\pi}(s,a)]
\]`

其中的`\(Q^{\pi}(s,a)\)`是状态-行为值函数。可见，策略梯度是关于状态和动作的期望，在求期望时，需要对状态分布和动作分布求积分，需要在状态空间和动作空间内大量采样，这样求出来的均值才能近似期望。

而确定性策略的动作是确定的，所以，如果存在确定性策略梯度，其求解**不需要在动作空间采样**，所以需要的样本数更少。对于动作空间很大的智能体（如多关节机器人），动作空间维数很大，有优势。

+ 随机策略的优点：随机策略可以将**探索和改善**集成到**一个策略**中

随机策略本身自带探索，可以通过探索产生各种数据（有好有坏），好的数据可以让强化学习算法改进当前策略。

而确定性策略给定状态和策略参数时，动作是固定的，所以无法探索其他轨迹或者访问其他状态。

**确定性策略**无法探索环境，所以需要通过**异策略**（off-policy）方法来进行学习，即行动策略和评估策略不是同一个策略。**行动策略采用随机策略，而评估策略要用确定性策略**。而整个确定性策略的学习框架采用的是AC方法。

## 3. AC框架

这里会参考[https://blog.csdn.net/jinzhuojun/article/details/72851548](https://blog.csdn.net/jinzhuojun/article/details/72851548)

Actor-Critic（AC）方法其实是policy-based和value-based方法的结合。因为它本身是一种PG方法，同时又结合了value estimation方法，所以有些地方将之归为PG方法的一种，有些地方把它列为policy-based和value-based以外的另一种方法。

+ Actor指的是行动策略，负责policy gradient学习策略
+ Critic指的是评估策略，负责policy evaluation估计value function

所以，

+ 一方面actor学习策略，而策略更新依赖critic估计的value function；
+ 另一方面critic估计value function，而value function又是策略的函数。

如果是Actor-only，那就是policy gradient，而如果是Critic-only，那就是Q-learning。

### 3.0 pg

[https://www.jianshu.com/p/2ccbab48414b](https://www.jianshu.com/p/2ccbab48414b)

[https://blog.csdn.net/qq_30615903/article/details/80747380](https://blog.csdn.net/qq_30615903/article/details/80747380)

如果一个动作得到的reward多，那么我们就使其出现的概率增加，如果一个动作得到的reward少，我们就使其出现的概率减小。

根据这个思想，我们构造如下的损失函数：

`\[
loss= -log(prob)*v_t
\]`

上式中log(prob)表示在状态s对所选动作a的吃惊度, 如果概率越小, 反向的log(prob)反而越大. 而vt代表的是当前状态s下采取动作a所能得到的奖励，这是当前的奖励和未来奖励的贴现值的求和。也就是说，我们的策略梯度算法必须要完成一个完整的eposide才可以进行参数更新，而不是像值方法那样，每一个(s,a,r,s')都可以进行参数更新。如果在prob很小的情况下, 得到了一个大的Reward, 也就是大的vt, 那么`\(log(prob)*v_t\)`就更大, 表示更吃惊, (我选了一个不常选的动作, 却发现原来它能得到了一个好的reward, 那我就得对我这次的参数进行一个大幅修改)。

Policy Gradient的核心思想是更新参数时有两个考虑：如果这个回合选择某一动作，下一回合选择该动作的概率大一些，然后再看奖惩值，如果奖惩是正的，那么会放大这个动作的概率，如果奖惩是负的，就会减小该动作的概率。

注意点：

+ 算法输出的是动作的概率，而不是Q值。
+ 损失函数的形式为：`\(loss= -log(prob)*v_t\)`
+ 需要一次完整的episode才可以进行参数的更新


### 3.1 随机策略AC方法

随机策略的梯度为

`\[
\triangledown _{\theta}J(\pi _{\theta})=E_{s\sim \rho ^{\pi},a\sim \pi_{\theta}}[\triangledown _{\theta}log\pi_{\theta}(a|s)Q^{\pi}(s,a)]
\]`

其中Actor方法用来调整`\(\theta\)`值，

Critic方法逼近值函数`\(Q^{w}(s,a)\approx Q^{\pi}(s,a)\)`，其中`\(w\)`为待逼近的参数，可以用TD学习的方法来评估值函数。

**异策略**随机梯度为

`\[
\triangledown _{\theta}J_{\beta}(\pi _{\theta})=E_{s\sim \rho ^{\pi},a\sim \beta}[\frac{\pi_{\theta}(a|s)}{\beta_{\theta}(a|s)}\triangledown _{\theta}log\pi_{\theta}(a|s)Q^{\pi}(s,a)]
\]`

和原公式的区别在于**采样策略为`\(\beta\)`**，即`\(a\sim \beta\)`，与行动策略不同，所以叫异策略。从而，多了一项`\(\frac{\pi_{\theta}(a|s)}{\beta_{\theta}(a|s)}\)`。

### 3.2 确定性策略AC方法（DPG）

确定性的策略梯度为：

`\[
\triangledown _{\theta}J(\mu _{\theta})=E_{s\sim \rho ^{\mu}}[\triangledown _{\theta}\mu_{\theta}(s)\triangledown _{a}Q^{\mu}(s,a)|_{a=\mu_{\theta}(s)}]
\]`

可见，区别如下：

+ `\(\pi_{\theta}\)`变成了`\(\mu_{\theta}\)`
+ 原来的`\(Q^{\pi}(s,a)\)`改成了`\(Q^{\mu}(s,a)|_{a=\mu_{\theta}(s)}\)`
+ 原来的`\(s\sim \rho ^{\pi}\)`变成了`\(s\sim \rho ^{\mu}\)`
+ 去掉了对于动作的采样`\(a\sim \pi _{\theta}\)`，而改成确定性的动作`\(a=\mu_{\theta}(s)\)`
+ 原来对`\(\pi\)`的梯度，即`\(\triangledown _{\theta}log\pi_{\theta}(a|s)\)`改成了对`\(\mu\)`的梯度`\(\triangledown _{\theta}\mu_{\theta}(s)\)`
+ 对于`\(Q\)`也要求一次关于`\(a\)`的梯度，即：`\(\triangledown _{a}Q^{\mu}(s,a)|_{a=\mu_{\theta}(s)}\)`，即回报函数对动作的导数

所以**异策略**确定性策略梯度为

`\[
\triangledown _{\theta}J_{\beta}(\mu _{\theta})=E_{s\sim \rho ^{\beta}}[\triangledown _{\theta}\mu_{\theta}(s)\triangledown _{a}Q^{\mu}(s,a)|_{a=\mu_{\theta}(s)}]
\]`

与异策略的随机策略梯度进行对比，可以发现少了重要性权重，即`\(\frac{\pi_{\theta}(a|s)}{\beta_{\theta}(a|s)}\)`。因为重要性采样是用简单的概率分布估计复杂的概率分布，而确定性策略的动作是确定值；

此外，确定性策略的值函数评估用的是Q-learning方法，也就是用TD(0)估计动作值函数，并且忽略重要性权重。

然后看一下确定性策略异策略AC算法的更新过程：

`\[
\begin{matrix}
\delta _t=r_t+ \gamma Q^{w}(s_{t+1},\mu_{\theta}(s_{t+1}))-Q^{w}(s_t,a_t)\\ 
w_{t+1}=w_t+\alpha _w\delta_t\triangledown _wQ^w(s_t,a_t)\\ 
\theta _{t+1}=\theta _t+\alpha _\theta \triangledown _{\theta} \mu _{\theta}(s_t)\triangledown _aQ^w(s_t,a_t)|_{a=\mu_{\theta}(s)}
\end{matrix}
\]`

前两行是利用值函数逼近的方法更新值函数参数`\(w\)`，使用的是TD，用Q-learning。

第3行是用确定性策略梯度方法更新策略参数`\(\theta\)`

### 3.3 深度确定性策略梯度方法（DDPG）

[Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971)

DDPG是深度确定性策略，复用DNN逼近行为值函数`\(Q^w(s,a)\)`和确定性策略`\(\mu_\theta (s)\)`。

在讲DQN时，当利用DNN进行函数逼近时，强化学习算法常常不稳定。因为训练nn时往往假设输入数据是独立同分布的，而强化学习的数据 是顺序采集的，数据间存在马尔科夫性，所以这些数据并非独立同分布。

为了打破数据间的相关性，DQN使用了两个技巧，经验回放和独立的目标网络。

DDPG就是将这两个技巧用到DPG算法中，DDPG的经验回放和DQN完全相同，这里介绍DDPG中的独立目标网络。

DDPG的目标值是上式中第一行的前两项，即

`\[
r_t+ \gamma Q^{w}(s_{t+1},\mu_{\theta}(s_{t+1}))
\]`

而所谓的独立目标网络，就是将上式的`\(w\)`和`\(\theta\)`单独拿出来，利用独立的网络对其进行更新，所以DDPG的更新公式为：

`\[
\begin{matrix}
\delta _t=r_t+ \gamma Q^{w^-}(s_{t+1},\mu_{\theta^-}(s_{t+1}))-Q^{w}(s_t,a_t)\\ 
w_{t+1}=w_t+\alpha _w\delta_t\triangledown _wQ^w(s_t,a_t)\\ 
\theta _{t+1}=\theta _t+\alpha _\theta \triangledown _{\theta} \mu _{\theta}(s_t)\triangledown _aQ^w(s_t,a_t)|_{a=\mu_{\theta}(s)}
\\ \theta^-=\tau \theta +(1-\tau)\theta^-
\\w^-=\tau w+(1-\tau)w^-
\end{matrix}
\]`

DDPG的整体流程如下：

>1. 使用权重`\(\theta ^Q\)`随机初始化critic网络`\(Q(s,a|\theta ^Q)\)`，使用权重`\(\theta ^{\mu}\)`随机初始化actor`\(\mu (s|\theta ^{\mu})\)`
>1. 使用权重`\({\theta ^{Q'}} \leftarrow \theta ^Q\)`初始化目标网络`\(Q'\)`，使用权重`\({\theta ^{\mu'}} \leftarrow \theta ^{\mu}\)`初始化`\(\mu'\)`
>1. 初始化replay buffer `\(R\)`
>1. For `\(episode = [1,...,M]\)` do
>    1. 初始化一个随机过程`\(\mathcal {N}\)`，即noise，以用于action exploration
>    1. 获取初始化的可观测状态`\(s_1\)`
>    1. For `\(t=[1,...T]\)` do
>        1. 根据当前的policy以用exploration noise，选择动作`\(a_t=\mu(s_t|\theta^{\mu})+\mathcal {N}_t\)`【这里体现了随机策略作为行动策略】
>        1. 执行动作`\(a_t\)`，得到回报`\(r_t\)`以及新的状态`\(s_{t+1}\)`
>        1. 将transition `\((s_t,a_t,r_t,s_{t+1})\)`存入`\(R\)`。
>        1. 从`\(R\)`中随机sample出一个minibatch(`\(N\)`个)的transitions，`\((s _i,a_i,r_i,s_{i+1})\)`
>        1. 令`\(y_i=r_i+\gamma {Q'}{(s_{i+1},{\mu'}(s_{i+1}|\theta ^{\mu'})|\theta ^{Q'}})\)`【即使用两个目标网络得predict的值`\(y_i\)`】
>        1. 通过最小化loss`\(L=\frac{1}{N}\sum_i(y_i-Q(s_i,a_i|\theta ^Q))^2\)`对critic `\(Q\)`进行更新
>        1. 通过采样的梯度，对actor policy`\(\mu\)`进行更新：
>        `\[\triangledown _{\theta ^\mu} {J}\approx \frac{1}{N}\sum_i\triangledown_aQ(s,a|\theta ^Q)|_{s=s_i,a=\mu(s_i)}\triangledown _{\theta ^\mu} {\mu(s|\theta ^\mu)|_{s_i}}\]`
>        1. 更新critic的目标网络`\(Q'\)`和actor的目标网络`\(\mu'\)`： 
>        `\[\begin{matrix}
\theta^{Q'}\leftarrow\tau \theta ^Q +(1-\tau)\theta^{Q'}
\\\theta^{\mu'}\leftarrow\tau \theta ^{\mu}+(1-\tau)\theta^{\mu'}
\end{matrix}\]`
>    1. End For
> 1. End For

注：

+ critic是`\(Q\)`，critic的目标网络是`\(Q'\)`
+ actor是`\(\mu\)`，actor的目标网络是`\(\mu'\)`
+ critic的参数`\(\theta ^Q\)`就是前面讲的`\(w\)`
+ critic的目标网络的参数`\(\theta ^{Q'}\)`就是前面讲的`\(w^-\)`
+ actor的参数`\(\theta ^\mu\)`就是前面讲的`\(\theta\)`
+ actor的目标网络的参数`\(\theta ^{\mu'}\)`就是前面讲的`\(\theta^-\)`

来看看ddpg的代码：[https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/9_Deep_Deterministic_Policy_Gradient_DDPG/DDPG_update.py](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/9_Deep_Deterministic_Policy_Gradient_DDPG/DDPG_update.py)

代码里有几个点可以注释下咯：

+ 需要求导的参数定义

```python
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')
```

+ actor和critic的定义

```python
        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True) # 是用来训练的
            a_ = self._build_a(self.S_, scope='target', trainable=False) # 目标网络只是隔tau后会直接更新参数
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True) # 是用来训练的
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False) # 目标网络只是隔tau后会直接更新参数
```

actor网络，输入状态`\(s\)`，输出动作`\(a\)`，由于是连续动作空间，所以`\(a\)`是一个```a_dim```维的向量，在tanh后，是-1到1之间，乘一个```a_bound```把输出值缩放到正确的值域里。

```python
    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')
```

critic网络，输入有两个参数，状态`\(s\)`和动作`\(a\)`，输出q值(是一个数字)：

```python
    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)
```

+ critic的loss：当前reward加上gamma乘以目标critic网络输出的q，是我们的td目标（即上面提到的`\(y_i\)`），而critic网络输出的q，则是我们当前critic网络的输出，两者之差的mse就是td_error，而我们需要对critic的参数求导，所以```ar_list=self.ce_params```

```python
        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)
```

+ actor的loss：可以发现上面讲的`\(\triangledown_aQ(s,a|\theta ^Q)|_{s=s_i,a=\mu(s_i)}\triangledown _{\theta ^\mu} {\mu(s|\theta ^\mu)|_{s_i}}\)`，其实就是`\(\frac{\partial Q(s,a)}{\partial a} \frac{\partial a}{\partial \theta ^\mu}\)`，也就是`\(\frac{\partial Q(s,a)}{\partial \theta ^\mu}\)`，所以就是以critic网络的输出q，对actor的参数进行求导，所以```var_list=self.ae_params```

```python
        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)
```

+ 更新目标网络的操作

```python
        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]
```

### 3.3 A3C(asynchronous advantage actor-critic)

详见：[https://daiwk.github.io/posts/rl-distributed-rl.html#2-a3c](https://daiwk.github.io/posts/rl-distributed-rl.html#2-a3c)

