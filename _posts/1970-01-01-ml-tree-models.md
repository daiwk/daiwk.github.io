---
layout: post
category: "knowledge"
title: "树模型"
tags: [gbdt, id3, c4.5, cart, ensemble, ]

---

目录

<!-- TOC -->

- [基础树模型](#%E5%9F%BA%E7%A1%80%E6%A0%91%E6%A8%A1%E5%9E%8B)
  - [ID3](#id3)
  - [C4.5](#c45)
  - [CART](#cart)
- [ensemble](#ensemble)
  - [ensemble的背景](#ensemble%E7%9A%84%E8%83%8C%E6%99%AF)
    - [1.模型选择（Model Selection）](#1%E6%A8%A1%E5%9E%8B%E9%80%89%E6%8B%A9model-selection)
    - [2.数据集过小或过大（Too much or too little data）](#2%E6%95%B0%E6%8D%AE%E9%9B%86%E8%BF%87%E5%B0%8F%E6%88%96%E8%BF%87%E5%A4%A7too-much-or-too-little-data)
    - [3.分治（Divide and Conquer）](#3%E5%88%86%E6%B2%BBdivide-and-conquer)
    - [4.数据融合（Data Fusion）](#4%E6%95%B0%E6%8D%AE%E8%9E%8D%E5%90%88data-fusion)
  - [boosting](#boosting)
  - [bagging](#bagging)
- [gbdt](#gbdt)
- [xgboost](#xgboost)

<!-- /TOC -->


## 基础树模型

### ID3

### C4.5

### CART

CART，又名分类回归树，是在ID3的基础上进行优化的决策树，学习CART记住以下几个关键点：

+ CART既能是分类树，又能是分类树；
+ 节点分裂的依据：
    + 分类树时，采用GINI值；
    + 回归树时，采用样本的最小方差；
+ CART是一棵二叉树

详见[https://www.cnblogs.com/canyangfeixue/p/7802835.html](https://www.cnblogs.com/canyangfeixue/p/7802835.html)



## ensemble

参考[https://blog.csdn.net/u010659278/article/details/44527437](https://blog.csdn.net/u010659278/article/details/44527437)

集成学习是指将**若干弱分类器组合之后产生一个强分类器**。弱分类器（weak learner）指那些分类准确率只稍好于随机猜测的分类器（error rate < 50%）。

集成算法成功的关键在于能**保证弱分类器的多样性**。

### ensemble的背景

参考[http://www.scholarpedia.org/article/Ensemble_learning](http://www.scholarpedia.org/article/Ensemble_learning)

#### 1.模型选择（Model Selection）


#### 2.数据集过小或过大（Too much or too little data）


#### 3.分治（Divide and Conquer）


#### 4.数据融合（Data Fusion） 



### boosting




### bagging






## gbdt

gbdt 是通过采用**加法模型**（即**对基函数进行线性组合**），以及不断减小训练过程产生的残差来达到将数据分类或者回归的算法。

参考《统计学习方法》P147-


参考[https://www.cnblogs.com/pinard/p/6140514.html](https://www.cnblogs.com/pinard/p/6140514.html)


参考[https://www.zybuluo.com/yxd/note/611571](https://www.zybuluo.com/yxd/note/611571)


参考[https://zhuanlan.zhihu.com/p/29765582](https://zhuanlan.zhihu.com/p/29765582)

再看一个很6的pdf：[https://daiwk.github.io/assets/gbdt.pdf](https://daiwk.github.io/assets/gbdt.pdf)


## xgboost

xgboost的树的叶子和predict结果有什么关系

+ 假设只有一棵树，
  + 如果是回归，那就是叶子的得分+0.5的bias。[https://github.com/dmlc/xgboost/issues/1077](https://github.com/dmlc/xgboost/issues/1077)
  + 如果是分类，那是叶子的得分sigmoid
+ 假设有多棵树，那分类就是每棵树的叶子的得分加起来再过个sigmoid

[https://stats.stackexchange.com/questions/395697/what-is-an-intuitive-interpretation-of-the-leaf-values-in-xgboost-base-learners](https://stats.stackexchange.com/questions/395697/what-is-an-intuitive-interpretation-of-the-leaf-values-in-xgboost-base-learners)


xgboost的predict函数线程不安全：[https://github.com/dmlc/xgboost/issues/311](https://github.com/dmlc/xgboost/issues/311)

解决：可以将handle放到一个pool里，每次要predict的时候，从pool里去拿。

xgboost调参：[https://segmentfault.com/a/1190000014040317](https://segmentfault.com/a/1190000014040317)

注意，```grid_scores_```这个参数改名了，改成了````cv_results_```。

xgboost的坑：

[https://xgboost.readthedocs.io/en/latest/faq.html#why-do-i-see-different-results-with-sparse-and-dense-data](https://xgboost.readthedocs.io/en/latest/faq.html#why-do-i-see-different-results-with-sparse-and-dense-data)

所以，如果我们用libsvm的格式来给python喂数据，如果用的是tree，那么需要加上missing参数：

```python
dtrain = xgb.DMatrix("./output/%sins.dat.train.cls" % (prefix) + "." + flag, missing=0.0)
dtest = xgb.DMatrix("./output/%sins.dat.test.cls" % (prefix) + "." + flag, missing=0.0)

params = {'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 5,
        'min_child_weight': 1,
        'eta': 0.1,
        'nthread': 8,
        'silent': 1}
```

而在线预测的时候，默认值给0.0：

```c++

int sample_rows = 1;
int cols = feature_count;
float test[sample_rows][cols];
for (int i = 0; i < sample_rows; i++) {
    for (int j = 0; j < cols; j++) {
        test[i][j] = 0;
    }
}

DMatrixHandle h_test;

// 这里的0.0就是Missing的默认值
int ret_x = XGDMatrixCreateFromMat((float *) test, sample_rows, cols, 0.0, &h_test);
if (ret_x < 0) {
    XGDMatrixFree(h_test);
    LOG(WARNING) << "fail in XGDMatrixCreateFromMat";
    return ret;
}
bst_ulong out_len;
const float * pred_result = nullptr;

std::shared_ptr<BoosterHandlePool> handle_pool = xgb_dict->get_dict();
if (handle_pool == nullptr) {
    return ret;
}
auto obj = handle_pool->get();
BoosterHandle *handle = obj.get();
if (handle == nullptr) {
    return ret;
}
int ret_p = XGBoosterPredict(*handle, h_test, 0, 0, &out_len, &pred_result);
if (ret_p) {
    pred_result = nullptr;
    LOG(WARNING) << "fail in XGBoosterPredict";
    return ret;
}

XGDMatrixFree(h_test);
```