---
layout: post
category: "dl"
title: "GAN-CDQN"
tags: [GAN-CDQN, 强化学习, 推荐, gan, cascade dqn, 级联dqn, ]
---

目录

<!-- TOC -->

- [简介](#简介)
- [Setting和RL Formulation](#setting和rl-formulation)
- [Generative Adversarial User Model](#generative-adversarial-user-model)
    - [User Behavior As Reward Maximization](#user-behavior-as-reward-maximization)
    - [Model Parameterization](#model-parameterization)
    - [Generative Adversarial Training](#generative-adversarial-training)
- [Cascading RL Policy for Recommendation](#cascading-rl-policy-for-recommendation)
    - [Cascading Q-Networks](#cascading-q-networks)
    - [Parameterization and Estimation](#parameterization-and-estimation)

<!-- /TOC -->


参考[ICML 2019 \| 强化学习用于推荐系统，蚂蚁金服提出生成对抗用户模型](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650763260&idx=3&sn=ae589196211189a8aba6f56a11e2cccb&chksm=871aab82b06d22942d37c9b6efe33cd9647050599f293e59ffc8ababeedb9f8c80afbd80b509&scene=0&xtrack=1&pass_ticket=TloMdmvUbLd5jnKvVTzrccQhGuskwL6KQ0HhJLF56Nwtcb16%2BVvMA09bw32tFrjs#rd)

[Generative Adversarial User Model for Reinforcement Learning Based Recommendation System](https://arxiv.org/pdf/1812.10613.pdf)

## 简介

本文提出利用**生成对抗网络**同时学习**用户行为模型transition**以及**奖励函数reward**。将该**用户模型**作为强化学习的**模拟环境**，研究者开发了全新的**Cascading-DQN**算法，从而得到了可以**高效处理大量候选**物品的**组合推荐**策略。

本文用真实数据进行了实验，发现和其它相似的模型相比，这一**生成对抗用户模型**可以**更好地解释用户行为**，而基于该模型的RL策略可以给用户带来**更好的长期收益**，并给系统提供**更高的点击率**。

RL在推荐场景中有以下问题：

+ 首先，驱动用户行为的兴趣点**（奖励函数）一般是未知的**，但它对于 RL 算法的使用来说至关重要。在用于推荐系统的现有RL算法中，奖励函数一般是**手动设计的（例如用 ±1 表示点击或不点击）**，这可能无法反映出用户对不同项目的偏好如何(如[Deep Reinforcement Learning for Page-wise Recommendations](https://arxiv.org/abs/1805.02343))。
+ 其次，**无模型**RL一般都需要**和环境（在线用户）进行大量的交互**才能学到良好的策略。但这在推荐系统设置中是不切实际的。如果推荐看起来比较随机或者推荐结果不符合在线用户兴趣，他会很快**放弃**!!这一服务。

为了解决**无模型**方法**样本复杂度大**的问题，**基于模型**的RL方法更为可取。近期有一些研究，在robotics applications中，在**相关但不相同**的**环境**设置中训练机器人策略，结果表明**基于模型**的RL**采样效率更高**。如[Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning](https://arxiv.org/abs/1708.02596)，还有[Gaussian Processes for Data-Efficient Learning in Robotics and Control](https://www.doc.ic.ac.uk/~mpd37/publications/pami_final_w_appendix.pdf)，还有[ Learning to adapt: Meta-learning for model-based control](https://arxiv.org/abs/1803.11347)。

基于模型的方法的优势在于可以**池化**大量的**off-policy数据**，而且可以用这些数据学习良好的环境动态模型，而无模型方法只能用**昂贵的on-policy**数据学习。但之前基于模型的方法一般都是根据物理或高斯过程设计的，而**不是根据用户行为的复杂序列**定制的。

本文的框架用统一的minimax框架学习**用户行为模型**和相关的**奖励函数**，然后再用这个模型**学习RL策略**。

主要贡献如下：

+ 开发了生成对抗学习（GAN）方法来对用户行为动态性(dynamics)建模，并recover奖励函数。可以通过**联合的minimax优化算法**同时评估这两个组件。该方法的优势在于：
    + 可以得到**更predictive(可预测的？)的用户模型**，而且可以用**与用户模型一致**的方法**学习奖励函数**；
    + 相较于手动设计的简单奖励函数，**从用户行为中学习到的奖励函数**更**有利于**后面的**强化学习**；
    + 学习到的用户模型使研究者能够为**新用户**执行基于模型的RL和在线适应从而实现更好的结果。
+ 用这一模型作为模拟环境，研究者还开发了**级联DQN**(cascade dqn)算法来获得组合推荐策略。**动作-值函数**的**级联设计**允许其在**大量候选物品**中找到要展示的物品的**最佳子集**，其**时间复杂度**和**候选物品的数量**呈**线性关系**，大大减少了计算难度。

用真实数据进行实验得到的结果表明，从保留似然性和点击预测的角度来说，这种生成对抗模型可以**更好地拟合用户行为**。根据学习到的用户模型和奖励，研究者发现评估推荐策略可以给用户带来**更好的长期累积奖励**。此外，在**模型不匹配**的情况下，基于模型的策略也能够**很快地适应新动态**（和无模型方法相比，**和用户交互的次数要少得多**）。

<html>
<br/>
<img src='../assets/gan-cdqn-illustration.png' style='max-height: 300px'/>
<br/>
</html>

图中绿线是推荐的信息流，黄线是用户的信息流。

## Setting和RL Formulation

setting：给用户展示了`\(k\)`个item，然后他点了**1个或者0个**，然后展示后`\(k\)`个item。

简单来讲，RL框架就是，推荐系统会在用户状态`\(\mathcal{s}\)`下，采用策略`\(\pi(\boldsymbol{s}, \mathcal{I})\)`来从集合`\(\mathcal{I}\)`中进行选择，使得如下的用户长期累积reward最大：

`\[
 \pi^{*}=\underset{\pi\left(\boldsymbol{s}^{t}, \mathcal{I}^{t}\right)}{\arg \max } \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^{t} r\left(\boldsymbol{s}^{t}, a^{t}\right)\right]
 \]`

其中，`\(s^{0} \sim p^{0}\)`，`\(\mathcal{A}^{t} \sim \pi\left(s^{t}, \mathcal{I}^{t}\right)\)`，`\(\boldsymbol{s}^{t+1} \sim P\left(\cdot | \boldsymbol{s}^{t}, \mathcal{A}^{t}\right)\)`，`\(a^{t} \in \mathcal{A}^{t}\)`。

+ 环境：在推荐的每一页的`\(k\)`个item中可以点击其中一个的用户
+ 状态`\(\boldsymbol{s}^{t} \in \mathcal{S}\)`：用户**历史点击**的一个**有序序列**
+ 动作`\(\mathcal{A}^{t} \in\left(\begin{array}{c}{\mathcal{I}^{t}} \\ {k}\end{array}\right)\)`：推荐系统从`\(\mathcal{I}^{t}\)`个候选中选择`\(k\)`个候选的**子集**。其中，`\(\left(\begin{array}{c}{\mathcal{I}^{t}} \\ {k}\end{array}\right)\)`表示`\(\mathcal{I}^{t}\)`中的所有的`\(k\)`元素子集。而`\(\mathcal{I}^{t} \subset \mathcal{I}\)`是所有候选`\(\mathcal{I}\)`在时间步`\(t\)`的候选子集。
+ 状态转移`\(P\left(\cdot | s^{t}, \mathcal{A}^{t}\right) : \mathcal{S} \times\left(\begin{array}{l}{\mathcal{I}} \\ {k}\end{array}\right) \mapsto \mathcal{P}(\mathcal{S})\)`：给定状态`\(\mathcal{s}^{t}\)`，以及展示的集合`\(\mathcal{A}^{t}\)`的情况下，转移到状态`\(\boldsymbol{s}^{t+1}\)`的转移概率。等价于[下文](#user-behavior-as-reward-maximization)提到的在用户行为上的分布`\(\phi\left(s^{t}, \mathcal{A}^{t}\right)\)`
+ 奖励函数`\(r\left(\boldsymbol{s}^{t}, \mathcal{A}^{t}, a^{t}\right) : \mathcal{S} \times\left(\begin{array}{l}{\mathcal{I}} \\ {k}\end{array}\right) \times \mathcal{I} \mapsto \mathbb{R}\)`：用户在状态`\(\boldsymbol{s}^{t}\)`下，采用动作`\(a^{t} \in \mathcal{A}^{t}\)`得到的回报。在这里假设推荐系统得到的回报和用户得到的回报一样，所以长期回报也是一样。
+ 策略`\(\mathcal{A}^{t} \sim \pi\left(s^{t}, \mathcal{I}^{t}\right) : \mathcal{S} \times 2^{\mathcal{I}} \mapsto \mathcal{P}\left(\left(\begin{array}{c}{\mathcal{I}_{k}} \\ {k}\end{array}\right)\right)\)`：在用户状态`\(\mathcal{s}^{t}\)`下，从集合`\(\mathcal{I}^{t}\)`中选择子集`\(\mathcal{A}^{t}\)`进行展示的概率。

可见，

+ **环境**、**状态**、**状态转移**与**用户**有关，
+ **行为**、**策略**与**推荐系统**有关。
+ **回报**与**推荐系统**和**用户**均有关。

使用`\(r\left(\boldsymbol{s}^{t}, \mathcal{A}^{t}, a^{t}\right)\)`来强调回报对推荐的action的依赖，也就是说，用户只能从展示的结果集中进行选择。其实，`\(r\left(\boldsymbol{s}^{t}, \mathcal{A}^{t}, a^{t}\right)=r\left(\boldsymbol{s}^{t}, a^{t}\right) \cdot \mathbf{1}\left(a^{t} \in \mathcal{A}^{t}\right)\)`。所以[下文](#user-behavior-as-reward-maximization)在讲用户模型的时候，就使用`\(r\left(\boldsymbol{s}^{t}, a^{t}\right)=r\left(\boldsymbol{s}^{t}, \mathcal{A}^{t}, a^{t}\right)\)`来表示了，假设`\(a^{t} \in \mathcal{A}^{t}\)`是true。

reward函数和transition都是未知的，可以从数据中学习。只要这两个学好了，那么就可以通过使用例如Q-learrning等算法，不断地对模型进行query，来估计上文提到的最优的策略`\( \pi^{*}=\underset{\pi\left(\boldsymbol{s}^{t}, \mathcal{I}^{t}\right)}{\arg \max } \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^{t} r\left(\boldsymbol{s}^{t}, a^{t}\right)\right]\)`。


## Generative Adversarial User Model

### User Behavior As Reward Maximization

### Model Parameterization

### Generative Adversarial Training

## Cascading RL Policy for Recommendation

### Cascading Q-Networks

### Parameterization and Estimation

