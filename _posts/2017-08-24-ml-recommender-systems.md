---
layout: post
category: "ml"
title: "推荐系统"
tags: [svd, svd++, als, rbm-cf, fm, eALS, ]
---

目录

<!-- TOC -->

- [ALS-WR](#als-wr)
- [SVD](#svd)
- [SVD++](#svd)
- [RBM-CF](#rbm-cf)
- [OCCF](#occf)
- [eALS](#eals)
    - [简介](#简介)
    - [related work](#related-work)
    - [预备知识](#预备知识)
        - [ALS](#als)
        - [使用Uniform Weighting来加速](#使用uniform-weighting来加速)
        - [generic element-wise ALS learner](#generic-element-wise-als-learner)

<!-- /TOC -->

[从item-base到svd再到rbm，多种Collaborative Filtering(协同过滤算法)从原理到实现](http://blog.csdn.net/dark_scope/article/details/17228643)


## ALS-WR

[Large-scale Parallel Collaborative Filtering for the Netflix Prize](https://endymecy.gitbooks.io/spark-ml-source-analysis/content/%E6%8E%A8%E8%8D%90/papers/Large-scale%20Parallel%20Collaborative%20Filtering%20the%20Netflix%20Prize.pdf)

目标是最小化：

`\[
f(U, M)=\sum_{(i, j) \in I}\left(r_{i j}-\mathbf{u}_{i}^{T} \mathbf{m}_{j}\right)^{2}+\lambda\left(\sum_{i} n_{u_{i}}\left\|\mathbf{u}_{i}\right\|^{2}+\sum_{j} n_{m_{j}}\left\|\mathbf{m}_{j}\right\|^{2}\right)
\]`

U矩阵的shape是`\(n_f\times n_u\)`，M矩阵的shape是`\(n_f\times n_m\)`

核心解法是：

+ M矩阵的第一行（也就是所有item的`\(n_f\)`个特征的第1个特征）初始化为该Item的平均分，其他位置设为一个比较小的随机值
+ 固定M，通过最小化目标函数求解U
+ 固定U，通过最小化目标函数求解M
+ 重复以上两步，直到满足终止条件

首先，当M固定时，来更新U。令u的导数为0，经过如下推导，可以得到

`\[\begin{aligned} & \frac{1}{2} \frac{\partial f}{\partial u_{k i}}=0, \quad \forall i, k \\ \Rightarrow & \sum_{j \in I_{i}}\left(\mathbf{u}_{i}^{T} \mathbf{m}_{j}-r_{i j}\right) m_{k j}+\lambda n_{u_{i}} u_{k i}=0, \quad \forall i, k \\ \Rightarrow & \sum_{j \in I_{i}} m_{k j} \mathbf{m}_{j}^{T} \mathbf{u}_{i}+\lambda n_{u_{i}} u_{k i}=\sum_{j \in I_{i}} m_{k j} r_{i j}, \quad \forall i, k \\ \Rightarrow &\left(M_{I_{i}} M_{I_{i}}^{T}+\lambda n_{u_{i}} E\right) \mathbf{u}_{i}=M_{I_{i}} R^{T}\left(i, I_{i}\right), \quad \forall i \\ \Rightarrow & \mathbf{u}_{i}=A_{i}^{-1} V_{i}, & \forall i \end{aligned}\]`

其中，

+ `\(A_{i}=M_{I_{i}} M_{I_{i}}^{T}+\lambda n_{u_{i}} E\)`
+ `\(V_{i}=M_{I_{i}} R^{T}\left(i, I_{i}\right)\)`
+ `\(E\)`是一个`\(n_{f} \times n_{f}\)`的identity矩阵(单位矩阵，对角线为1，其他位置是0)。
+ `\(M_{I_{i}}\)`是从`\(M\)`中把`\(j \in I_{i}\)`这些列选出来组成的矩阵
+ `\(R\left(i, I_{i}\right)\)`是取出`\(R\)`矩阵的第i行，然后再取出`\(j \in I_{i}\)`这些列得到的行向量

类似地，固定U来更新M：

`\[
\mathbf{m}_{j}=A_{j}^{-1} V_{j}, \quad \forall j
\]`

同理，

+ `\(A_{j}=U_{I_{j}} U_{I_{j}}^{T}+\lambda n_{m_{j}} E\)`
+ `\(V_{j}=U_{I_{j}} R\left(I_{j}, j\right)\)`
+ `\(U_{I_{j}}\)`是从`\(U\)`中把`\(i \in I_{j}\)`这些列取出来组成的矩阵
+ `\(R\left(I_{j}, j\right)\)`是取出`\(R\)`的第j列，然后再取出`\(i \in I_{j}\)`这些行组成的列向量

本文是用matlab来解的，了解一下求逆的时候用到的左除和右除。。参考[https://blog.csdn.net/qq_40655270/article/details/78250699](https://blog.csdn.net/qq_40655270/article/details/78250699)

左除：

`\(Ax=b\)`得到`\(x=A^{-1}b\)`。`\(A^{-1}\)`可以看成`\(1/A\)`，也就是说A在左边，而A是分母，所以，在matlab中写成```A\b```。

右除：

`\(xA=b\)`得到`\(x=bA^{-1}\)`。`\(A^{-1}\)`可以看成`\(1/A\)`，所以在matlab中写成```b/A```。

所以论文里的```X = matrix \ vector```就是matrix的逆乘以vector

## SVD

hinton 07年提出的[Restricted boltzmann machines for collaborative filtering](https://www.cs.toronto.edu/~rsalakhu/papers/rbmcf.pdf)里面提到了用sgd来训练svd：

`\[
\begin{aligned} f &=\sum_{i=1}^{N} \sum_{j=1}^{M} I_{i j}\left(\mathbf{u}_{\mathbf{i}} \mathbf{v}_{\mathbf{j}}^{\prime}-Y_{i j}\right)^{2} \\ &+\lambda \sum_{i j} I_{i j}\left(\left\|\mathbf{u}_{\mathbf{i}}\right\|_{F r o}^{2}+\left\|\mathbf{v}_{\mathbf{j}}\right\|_{F r o}^{2}\right) \end{aligned}
\]`

而在Koren的[Advances in collaborative filtering](https://datajobs.com/data-science-repo/Collaborative-Filtering-[Koren-and-Bell].pdf)有详细的介绍，其实就是加上一个user bias，再加一个item bias：

`\[
\hat{r}_{u i}=\mu+b_{i}+b_{u}+q_{i}^{T} p_{u}
\]`

优化目标就是：

`\[
\min _{b_{*}, q_{*}, p_{*}} \sum_{(u, i) \in \mathscr{K}}\left(r_{u i}-\mu-b_{i}-b_{u}-q_{i}^{T} p_{u}\right)^{2}+\lambda_{4}\left(b_{i}^{2}+b_{u}^{2}+\left\|q_{i}\right\|^{2}+\left\|p_{u}\right\|^{2}\right)
\]`

定义残差`\(e_{u i} \stackrel{\mathrm{def}}{=} r_{u i}-\hat{r}_{u i}\)`，使用梯度下降更新：

`\[
\begin{array}{l}{b_{u} \leftarrow b_{u}+\gamma \cdot\left(e_{u i}-\lambda_{4} \cdot b_{u}\right)} \\ {b_{i} \leftarrow b_{i}+\gamma \cdot\left(e_{u i}-\lambda_{4} \cdot b_{i}\right)} \\ {q_{i} \leftarrow q_{i}+\gamma \cdot\left(e_{u i} \cdot p_{u}-\lambda_{4} \cdot q_{i}\right)} \\ {p_{u} \leftarrow p_{u}+\gamma \cdot\left(e_{u i} \cdot q_{i}-\lambda_{4} \cdot p_{u}\right)}\end{array}
\]`


## SVD++

[Advances in collaborative filtering](https://datajobs.com/data-science-repo/Collaborative-Filtering-[Koren-and-Bell].pdf)

附：简单说一下svd++，就是user向量再加上这个用户的邻域信息(图中的y是用户的N(u)个历史item的隐式反馈)：

`\[
\hat{r}_{u i}=\mu+b_{i}+b_{u}+q_{i}^{T}\left(p_{u}+|\mathrm{R}(u)|^{-\frac{1}{2}} \sum_{j \in \mathrm{R}(u)} y_{j}\right)
\]`

梯度下降：

`\[
\begin{array}{l}{b_{u} \leftarrow b_{u}+\gamma \cdot\left(e_{u i}-\lambda_{5} \cdot b_{u}\right)} \\ {b_{i} \leftarrow b_{i}+\gamma \cdot\left(e_{u i}-\lambda_{5} \cdot b_{i}\right)} \\ {q_{i} \leftarrow q_{i}+\gamma \cdot\left(e_{u i} \cdot\left(p_{u}+|\mathrm{R}(u)|^{-\frac{1}{2}} \sum_{j \in \mathrm{R}(u)} y_{j}\right)-\lambda_{6} \cdot q_{i}\right)} \\ {p_{u} \leftarrow p_{u}+\gamma \cdot\left(e_{u i} \cdot q_{i}-\lambda_{6} \cdot p_{u}\right)} \\ {\forall j \in \mathrm{R}(u) :} \\ {y_{j} \leftarrow y_{j}+\gamma \cdot\left(e_{u i} \cdot|\mathrm{R}(u)|^{-\frac{1}{2}} \cdot q_{i}-\lambda_{6} \cdot y_{j}\right)}\end{array}
\]`

## RBM-CF

hinton 07年提出的[Restricted boltzmann machines for collaborative filtering](https://www.cs.toronto.edu/~rsalakhu/papers/rbmcf.pdf)

## OCCF

[One-Class Collaborative Filtering](http://www.rongpan.net/publications/pan-oneclasscf.pdf)

对missing data有两种权重策略：

+ AMAU：All missing as unknown，观测到的数据的权重置成1，missing data的权重全部设成0。
+ AMAN：All missing as negative。观测到的数据的权重置成1，missing data的权重全部设成1。（论文里没直接提到，不过好像是这么个意思。。）

wALS的损失函数就是要最小化：

`\[
\begin{aligned} \mathcal{L}(\boldsymbol{U}, \boldsymbol{V})=& \sum_{i j} W_{i j}\left(R_{i j}-\boldsymbol{U}_{i .} \boldsymbol{V}_{j .}^{T}\right)^{2} \\ &+\lambda\left(\|\boldsymbol{U}\|_{F}^{2}+\|\boldsymbol{V}\|_{F}^{2}\right) \end{aligned}
\]`

固定V，对U求导，令导数为0，去更新U的时候，推了半天，得到公式(6)

`\[
\begin{array}{r}{\boldsymbol{U}_{i .}=\boldsymbol{R}_{i .} \widetilde{\boldsymbol{W}_{i .}} \boldsymbol{V}\left(\boldsymbol{V}^{T} \widetilde{\boldsymbol{W}_{i .}} \boldsymbol{V}+\lambda\left(\sum_{j} W_{i j}\right) \boldsymbol{I}\right)^{-1}} \\ {\forall 1 \leq i \leq m}\end{array}
\]`

其中，`\(\widetilde{\boldsymbol{W}_{i .}} \in \mathfrak{R}^{n \times n}\)`是一个对角矩阵，对角线元素为`\(\boldsymbol{W}_{i .}\)`，而`\(I\)`是一个`\(d\times d\)`的单位矩阵（主对角线全1，其他全0）

而固定U，对V求导，令导数为0，更新V，得到公式(7)：

`\[
\begin{array}{r}{\boldsymbol{V}_{j .}=\boldsymbol{R}_{. j}^T \widetilde{\boldsymbol{W}_{. j}} \boldsymbol{U}\left(\boldsymbol{U}^{T} \widetilde{\boldsymbol{W}_{. j}} \boldsymbol{U}+\lambda\left(\sum_{j} W_{i j}\right) \boldsymbol{I}\right)^{-1}} \\ {\forall 1 \leq j \leq n}\end{array}
\]`

类似地，`\(\widetilde{\boldsymbol{W}_{. j}} \in \mathfrak{R}^{m \times m}\)`是一个对角矩阵，对角线元素为`\(\boldsymbol{W}_{. j}\)`。

然后整个wALS的过程就是不断地交替更新U和V直到满足收敛条件。

注意，这里的权重不是可训练的参数，是一开始以某种方式进行分配的，论文给出如下三种权重分配方式：

正样本，全部都是`\(W_{i j}=1\)`

对于missing data：

+ Uniform方式：`\(W_{i j}=\delta\)`，其中`\(\delta \in[0,1]\)`，每个missing data采用同一个权重
+ User-Oriented方式：`\(W_{i j} \propto \sum_{j} R_{i j}\)`。相当于对于一个用户来讲，如果他的正例比较多，说明他已经表达了很充分『什么样的东西他喜欢』了，那么他的missing item是负样本的概率更大
+ Item-Oriented方式：`\(W_{i j} \propto m-\sum_{i} R_{i j}\)`，如果一个item已经被很多用户喜欢了，那么它成为负样本的概率就更小了

## eALS

[Fast Matrix Factorization for Online Recommendation with Implicit Feedback](https://arxiv.org/pdf/1708.05024.pdf)

### 简介

以往的MF模型对于missing data，都是直接使用uniform weight(均匀分布)。然而在真实场景下，这个均匀分布的假设往往是不成立的。而且很多offline表现好的，到了动态变化的online场景上，往往表现不好。

文章使用**item的popularity**来给missing data权重，并提出了element-wise Alternating Least Squares(**eALS**)，来对missing data的权重是变量的问题进行学习。对于新的feedback，设计了增量更新的策略。对于两个offline和online的公开数据集，eALS都能取得比sota的隐式MF方法更好的效果。

之前的MF大多关注显式反馈，也就是用户的打分行为直接表达了用户对item的喜好程度。这种建模方式是基于这样一种假设：**大量的无标注的ratings(例如，missing data)与用户的preference无关**。这就大大减轻了建模的工作量，大量类似的复杂模型被提出了，例如SVD++、time-SVD等。

但在实际应用中，用户往往有大量的隐式反馈，例如浏览历史、购买历史等，而负反馈却是稀缺的。如果只对正反馈建模，那得到的是用户profile的一种biased的表示。针对这种负反馈缺失的场景（也叫one-class problem，参考[One-class collaborative filtering](http://www.rongpan.net/publications/pan-oneclasscf.pdf)），比较流行的解法是把missing data当做负反馈。但如果观测数据和missing data都要考虑的话，算法的学习效率就会大大降低。而且现实场景中，新的user/item/interaction都是流式地不停地进来的，所以模型的快速更新就更重要了。

针对隐式反馈和online learning这两大问题，KDD2015的[Dynamic matrix factorization with priors on unknown values](https://arxiv.org/abs/1507.06452)已经提出了一种方法**Dynamic MF**，但本文作者认为，这种方法对missing data的建模是『unrealistic』且『suboptimal』 的。也就是这种方法对missing data给予了uniform的权重，认为所有的missing data有同等概率被视为负反馈。另外这篇文章使用的是梯度下降，需要expensive line search来找到每一步的最优学习率。

本文的方法比Huyifan和Koren在[Collaborative filtering for implicit feedback datasets](http://yifanhu.net/PUB/cf.pdf)中用的ALS要快K倍，K是latent factor的num，这个速度和上面提到的**Dynamic MF**一样快。本文还提出了针对新数据的增量更新策略，能够快速更新模型参数。另外本文提出的方法不需要学习率，所以不需要像sgd那样去调学习率。

### related work

针对负反馈缺失的问题，有以下两种策略：

+ sample based learning：从missing data中采样出一部分，当成负反馈。性能好，但风险是效果不一定好。
+ whole-data based learning：所有missing data都当做是负反馈。覆盖会更多，但性能是瓶颈。

只有[Mind the gaps: Weighting the unknown in large-scale one-class collaborative filtering](http://agents.sci.brooklyn.cuny.edu/internal/proceedings/kdd/kdd2009/docs/p667.pdf)和[One-class collaborative filtering](http://www.rongpan.net/publications/pan-oneclasscf.pdf)考虑了非uniform的weight方法。但时间复杂度较高，没法在大规模的数据集上使用。

而优化方法方面，Koren的[Collaborative filtering with temporal dynamics](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.379.1951&rep=rep1&type=pdf)和[Bpr: Bayesian personalized ranking from implicit feedback](https://arxiv.org/pdf/1205.2618.pdf)使用的是SGD，而[Dynamic matrix factorization with priors on unknown values](https://arxiv.org/abs/1507.06452)和[Discrete collaborative filtering](https://dl.acm.org/citation.cfm?id=2911502)使用的是Coordinate Descent (CD)，而[Fast context-aware recommendations with factorization machines](https://dl.acm.org/citation.cfm?id=2010002&dl=ACM&coll=DL)使用的是Markov Chain Monto Carlo(MCMC)。

SGD是最流行的，但对于[Collaborative filtering for implicit feedback datasets](http://yifanhu.net/PUB/cf.pdf)提到的whole-data based MF来说，是不适合的，因为需要考虑整个user-item的交互矩阵，训练样本特别多。所以对于这种问题，ALS是CD的一个instantiation(实例化)，在[Collaborative filtering for implicit feedback datasets](http://yifanhu.net/PUB/cf.pdf)、[Mind the gaps: Weighting the unknown in large-scale one-class collaborative filtering](http://agents.sci.brooklyn.cuny.edu/internal/proceedings/kdd/kdd2009/docs/p667.pdf)、[One-class collaborative filtering](http://www.rongpan.net/publications/pan-oneclasscf.pdf)、[Training and testing of recommender systems on data missing not at random](https://dl.acm.org/citation.cfm?id=1835895)中都提到了。而这种方法时间复杂度较高，在[Fast als-based matrix factorization for explicit and implicit feedback datasets](https://dl.acm.org/citation.cfm?id=1864726)和[Effective latent models for binary feedback in recommender systems](http://www.cs.toronto.edu/~mvolkovs/sigir2015_svd.pdf)中都提到了。而[Fast als-based matrix factorization for explicit and implicit feedback datasets](https://dl.acm.org/citation.cfm?id=1864726)提到了als的一种近似解法；而[Dynamic matrix factorization with priors on unknown values](https://arxiv.org/abs/1507.06452)使用的是Randomized block Coordinate Descent (RCD)；而[Effective latent models for binary feedback in recommender systems](http://www.cs.toronto.edu/~mvolkovs/sigir2015_svd.pdf)则通过neighbor-based similarly来丰富隐式反馈矩阵，然后使用unweighted SVD。

而推荐系统的增量更新这块，有这么些研究：

+ neighbor-based：[TencentRec: Real-time stream recommendation in practice](http://net.pku.edu.cn/~cuibin/Papers/2015SIGMOD-tencentRec.pdf)
+ graph-based：[Trirank: Review-aware explainable recommendation by modeling aspects](https://www.comp.nus.edu.sg/~kanmy/papers/cikm15-trirank-cr.pdf)
+ probabilistic：[Google News Personalization: Scalable online collaborative filtering](https://www2007.org/papers/paper570.pdf)。用的plsi
+ MF：
  + [Dynamic matrix factorization with priors on unknown values](https://arxiv.org/abs/1507.06452)
  + [Real-time top-n recommendation in social streams](https://dl.acm.org/citation.cfm?id=2365968)
  + [Online learning for collaborative filtering](https://ieeexplore.ieee.org/document/6252670)
  + [Online-updating regularized kernel matrix factorization models for large scale recommender systems](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.165.8010&rep=rep1&type=pdf)

对MF来讲，在线更新有如下研究：

+ SGD：
  + [Real-time top-n recommendation in social streams](https://dl.acm.org/citation.cfm?id=2365968)
  + [Online-updating regularized kernel matrix factorization models for large scale recommender systems](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.165.8010&rep=rep1&type=pdf)
+ RCD：
  + [Dynamic matrix factorization with priors on unknown values](https://arxiv.org/abs/1507.06452)
+ dual-averaging：
  + [Online learning for collaborative filtering](https://ieeexplore.ieee.org/document/6252670)

本文是第一个用ALS的。

### 预备知识

#### ALS

假设

`\[
\hat{r}_{u i}=<\mathbf{p}_{u}, \mathbf{q}_{i}>=\mathbf{p}_{u}^{T} \mathbf{q}_{i}
\]`

考虑bias的话，就是

`\[
\hat{r}_{u i}=b_{u}+b_{i}+<\mathbf{p}_{u}^{B}, \mathbf{q}_{i}^{B}>
\]`

可以转化为`\(\mathbf{p}_{u} \leftarrow\left[\mathbf{p}_{u}^{B}, b_{u}, 1\right]\)`和`\(\mathbf{q}_{i} \leftarrow\left[\mathbf{q}_{i}^{B}, 1, b_{i}\right]\)`，这样就又是`\(\hat{r}_{u i}=<\mathbf{p}_{u}, \mathbf{q}_{i}>=\mathbf{p}_{u}^{T} \mathbf{q}_{i}\)`啦~~

目标函数是

`\[
J=\sum_{u=1}^{M} \sum_{i=1}^{N} w_{u i}\left(r_{u i}-\hat{r}_{u i}\right)^{2}+\lambda\left(\sum_{u=1}^{M}\left\|\mathbf{p}_{u}\right\|^{2}+\sum_{i=1}^{N}\left\|\mathbf{q}_{i}\right\|^{2}\right)
\]`

也就是要最小化

`\[
J_{u}=\left\|\mathbf{W}^{u}\left(\mathbf{r}_{u}-\mathbf{Q} \mathbf{p}_{u}\right)\right\|^{2}+\lambda\left\|\mathbf{p}_{u}\right\|^{2}
\]`

用ALS的话，得到的就是

`\[
\begin{aligned} \frac{\partial J_{u}}{\partial \mathbf{p}_{u}} &=2 \mathbf{Q}^{T} \mathbf{W}^{u} \mathbf{Q} \mathbf{p}_{u}-2 \mathbf{Q}^{T} \mathbf{W}^{u} \mathbf{r}_{u}+2 \lambda \mathbf{p}_{u}=0 \\ \Rightarrow \mathbf{p}_{u} &=\left(\mathbf{Q}^{T} \mathbf{W}^{u} \mathbf{Q}+\lambda \mathbf{I}\right)^{-1} \mathbf{Q}^{T} \mathbf{W}^{u} \mathbf{r}_{u} \end{aligned}
\]`

同样地，可以求出`\(\mathbf{q}_{i}\)`的解

复杂度的话，对一个`\(K \times K\)`的矩阵求逆是一个比较耗时的操作，复杂度是`\(O\left(K^{3}\right)\)`。所以更新一次user向量的复杂度是`\(O\left(K^{3}+N K^{2}\right)\)`。然后整个ALS的过程的复杂度就是`\(O\left((M+N) K^{3}+M N K^{2}\right)\)`。

#### 使用Uniform Weighting来加速

对`\(\mathbf{R}\)`中的所有0元素，给同样的权重`\(w_{0}\)`。就有：

`\[
\mathbf{Q}^{T} \mathbf{W}^{u} \mathbf{Q}=w_{0} \mathbf{Q}^{T} \mathbf{Q}+\mathbf{Q}^{T}\left(\mathbf{W}^{u}-\mathbf{W}^{0}\right) \mathbf{Q}
\]`

其中，`\(\mathbf{W}^{0}\)`是一个对角矩阵，对角线元素都是`\(w_{0}\)`。

因为`\(\mathbf{Q}^{T} \mathbf{Q}\)`和`\(u\)`无关，所以可以预先算好。

而`\(\mathbf{W}^{u}-\mathbf{W}^{0}\)`只有`\(\left|\mathcal{R}_{u}\right|\)`个非0值，所以上面这个式子的计算复杂度是`\(O\left(\left|\mathcal{R}_{u}\right| K^{2}\right)\)`

所以总的时间复杂度是`\(O\left((M+N) K^{3}+|\mathcal{R}| K^{2}\right)\)`

而SGD的时间复杂度是`\(O(|\mathcal{R}| K)\)`，远比ALS要小！！

#### generic element-wise ALS learner

在[Fast context-aware recommendations with factorization machines](https://dl.acm.org/citation.cfm?id=2010002&dl=ACM&coll=DL)里提到了

对`\(p_{u f}\)`求导

`\[
\frac{\partial J}{\partial p_{u f}}=-2 \sum_{i=1}^{N}\left(r_{u i}-\hat{r}_{u i}^{f}\right) w_{u i} q_{i f}+2 p_{u f} \sum_{i=1}^{N} w_{u i} q_{i f}^{2}+2 \lambda p_{u f}
\]`
