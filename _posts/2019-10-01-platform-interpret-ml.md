---
layout: post
category: "platform"
title: "InterpretML"
tags: [InterpretML, LIME, H2O, 可解释性, ]
---

目录

<!-- TOC -->

- [InterpretML](#interpretml)
- [其他库](#%e5%85%b6%e4%bb%96%e5%ba%93)
  - [LIME](#lime)
  - [RSLIME](#rslime)
  - [H2O](#h2o)
  - [ACE(cv相关)](#acecv%e7%9b%b8%e5%85%b3)

<!-- /TOC -->

## InterpretML

[InterpretML: A Unified Framework for Machine Learning Interpretability](https://arxiv.org/abs/1909.09223v1)

[https://github.com/microsoft/interpret](https://github.com/microsoft/interpret)

InterpretML 是一个为实践者和研究者提供机器学习可解释性算法的开源 Python 软件包。InterpretML 能提供以下两种类型的可解释性：（1）明箱（glassbox），这是针对可解释性设计的机器学习模型（比如线性模型、规则列表、广义相加模型）；（2）黑箱（blackbox）可解释技术，用于解释已有的系统（比如部分依赖、LIME）。这个软件包可让实践者通过在一个统一的 API 下，借助内置的可扩展可视化平台，使用多种方法来轻松地比较可解释性算法。InterpretML 也包含了可解释 Boosting 机（Explanable Boosting Machine，EBM）的首个实现，这是一种强大的可解释明箱模型，可以做到与许多黑箱模型同等准确的性能。


## 其他库

参考[吐血整理！绝不能错过的24个顶级Python库](https://zhuanlan.zhihu.com/p/76112940)

### LIME

[“Why Should I Trust You?” Explaining the Predictions of Any Classifier](https://arxiv.org/pdf/1602.04938.pdf)

[https://github.com/marcotcr/lime](https://github.com/marcotcr/lime)

LIME是一种算法（库），可以解释任何分类器或回归量的预测。LIME是如何做到的呢？通过可解释的模型在局部不断接近预测值，这个模型解释器可用于生成任何分类算法的解释。

[在机器学习模型中建立信任（在Python中使用LIME）](https://www.analyticsvidhya.com/blog/2017/06/building-trust-in-machine-learning-models/)

### RSLIME

[RSLIME: An Efficient Feature Importance Analysis Approach for Industrial Recommendation Systems](https://ieeexplore.ieee.org/document/8852034)

[为什么刷小视频停不下来？爱奇艺用这篇论文告诉你答案](https://mp.weixin.qq.com/s/mazNz9sGyxH-oASqg80Pgw)

相比于传统的视频推荐系统，爱奇艺的 UGC 推荐和小视频分发有四个极其困难的方面：

+ 新鲜度：爱奇艺的小视频应用的语料库非常动态，用户每天都会上传数十万条新视频。推荐系统应具有足够的响应能力，以便建模新上传的视频和最新的用户行为。
+ 冷启动：鉴于小视频有更高的及时性要求和更低的用户黏性，推荐系统面临着严重的用户和项目冷启动问题，这会有损基于协同过滤（CF）的方法的性能。
+ 多样性：由于视频类型和用户人口统计分布的多样性，爱奇艺的视频标签系统以及用户兴趣画像都比传统的视频推荐复杂得多，这也使得相关特征极其分散。内容和用户的多样性也会使得小视频推荐系统的结果不能稳健地应对输入中的错误。
+ 兴趣转移：历史用户行为并不总是可靠的。每位用户在一个小时内就可能浏览数十个小视频，他们感兴趣的内容也会发生巨大的变化。一旦用户对之前的视频感到厌烦，他们就会渴望探索新的类别。因此，把握短期和长期用户偏好之间的平衡是至关重要的。

针对这些难题，爱奇艺的研究者提出了一种遵循多阶段流程的模型，其由三个模块构成，即用户画像（User Profile）、召回（Recall）和排序 Ranking）。为了提升小视频推荐系统的表现，每个模块中都使用了广泛的模型集成方法。下面简要介绍了其系统结构：

+ 用户画像：对用户的人口统计属性、历史行为、兴趣和偏好的多维度分析。用户画像可用作实现个性化推荐的基石。
+ 召回：多种协同过滤（CF）算法（基于物品的 CF、基于用户的 CF、矩阵分解、Item2Vec 等）和多种基于内容的过滤（CBF）方法的组合。这些模型的结果会被聚合起来，为每个请求构建一个视频候选项语料库，其中通常包含数百条视频。
+ 排序：一个用于评估视频候选项的分数的点击率预估模型，然后将少量最佳推荐的视频推送到用户界面。

为了监控排序模型的工作方式是否有如预期以及是否能加速模型的迭代过程，爱奇艺提出了一种与模型无关的推荐系统局部可解释方法 Recommendation System Boosted Local Interpretable Model-Agnostic Explanations Method（RSLIME，），可为其排序模型提供特征重要度分析。RSLIME 有望为推荐系统中的特征选取过程提供参照，并帮助推荐系统开发者不必太过麻烦就能调整他们的排序模型。RSLIME 具有以下特点：

+ 对于单个输入样本，RSLIME 可以生成特征重要度的估计，而不管排序模块中所用的架构为何。然后可以基于这些特征重要度解释推荐结果。
+ 对于多个输入样本，RSLIME 可以结合多个样本的预测结果进行整体分析，并给出准确的特征重要度估计。
+ RSLIME 可对稀疏特征的影响进行高效的分析，从而指导模型的优化和特征的选择


其中：

+ DNN：DNN 使用的是一个带有三个隐藏层的全连接网络，其维度分别为 1024、512 和 256。DNN 的输入是用户和视频的预训练的特征嵌入，这基于用户行为和视频语义内容。爱奇艺的模型使用了 DNN 来提升排序模块的准确度以及在线 A/B 测试中的泛化能力。
+ GBDT：GBDT 是指多个决策树的基于提升（boosting）的集成。GBDT 的叶节点自动表示所选择的重要特征，其可被用于提升 FM 的性能。GBDT 先要单独进行预训练，之后才会与 FM 和 DNN 进行联合训练。由于 GBDT 对非归一化的特征的存在而言是稳健的，所以其输入中归一化和非归一化的稠密特征都可以有。
+ FM：FM 可自动执行特征组合和二阶交叉特征计算。因此，FM 可以执行特征融合和在 GBDT 输出和稀疏特征上执行交叉，由此能在推荐点击率（CTR）预估方面取得当前最佳的结果。
+ Sigmoid：Sigmoid 能为 DNN 和 FM 的输出加权并在其结果上执行 sigmoid 变换。


这里使用 X 表示输入数据，x 表示单个输入样本。样本 x 的 n 维输入特征表示为 Zn 或 (z1…zn)。z 表示单个特征组合，推理模型表示为 f。线性回归模型 g ∈ G 等可解释的模型经过训练后用于执行单个案例的特征重要度分析。

RSLIME 是局部可解释的与模型无关的解释（LIME）方法的一种扩展。LIME 使用了一种可解释的模型来评估推理模型 f 在单个输入样本 x 上的特征重要度。设有 100 个特征 (z1…z100) 的一个输入样本 x，f(x) 是 x 的推理结果。

LIME 首先会自动生成数千个不同的特征组合（比如 z1…z99、z2…z100）。然后对于每个特征组合，LIME 根据这个组合内部的特征（同时掩盖其它所有特征）计算该推理模型 f 的预测结果。为了说明清楚，使用特征组合 z1…z99 时，表示 z100 被 0 掩码掩盖。

给定每个特征组合 z 和对应的预测结果 f(z)，LIME 会计算 z 和 x 的距离并将其作为 z 的权重，然后会训练一个可解释的模型（以线性回归模型为例）g 将 z 映射到 f(z) 和 f(x) 之间的绝对差值，然后用作单个案例特征重要度的直接指示。这个可解释模型中每个特征的最终权重都会被用作特征重要度。因此，LIME 的局部特征诊断算法可以表示为：

`\[
\varepsilon(x)=\operatorname{argmin}_{g \in G} L\left(f, g, \pi_{x}(z)\right)+\Omega(g)
\]`

`\(\Omega(g)\)`表示模型复杂度，`\(\pi_{x}(z)\)`表示样本x的特征组合。

### H2O

[https://github.com/h2oai/mli-resources](https://github.com/h2oai/mli-resources)

H2O的无人驾驶AI，提供简单的数据可视化技术，用于表示高度特征交互和非线性模型行为，通过可视化提供机器学习可解释性（MLI），说明建模结果和模型中特征的影响。

[机器学习可解释性](https://www.h2o.ai/wp-content/uploads/2018/01/Machine-Learning-Interpretability-MLI_datasheet_v4-1.pdf)


### ACE(cv相关)

[Towards Automated Concept-based Explanation](https://arxiv.org/pdf/1902.03129.pdf)

[AI眼中的世界是什么样子？谷歌新研究找到了机器的视觉概念](https://mp.weixin.qq.com/s/JXJYvnLqLLuSclsd4ZTySA)

[https://github.com/amiratag/ACE](https://github.com/amiratag/ACE)
