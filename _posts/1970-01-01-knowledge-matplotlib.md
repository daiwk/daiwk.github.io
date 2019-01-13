---
layout: post
category: "knowledge"
title: "matplotlib"
tags: [matplotlib, ]
---

目录

<!-- TOC -->


<!-- /TOC -->

[50种常用的matplotlib可视化，再也不用担心模型背着我乱跑了](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650755443&idx=3&sn=0229c2d8eeb8e07486d6393cd9b983ae&chksm=871a950db06d1c1b9d7a08b38f2f16f974e5a79f994fb44175dab7a3ef5d7ece39e7763c8f49&mpshare=1&scene=1&srcid=0113CI7lPTEcV99uFQdNOXdp&pass_ticket=GbqnkzYDgSDQxJoviNYzckA8ZJ6bULsWpoyug4CHgCsT0B80C5nEC38bRj4CywCT#rd)

原始文章[https://www.machinelearningplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python](https://www.machinelearningplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python)

通用配置：

```python
# !pip install brewer2mpl
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')

large = 22; med = 16; small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")
%matplotlib inline

# Version
print(mpl.__version__)  #> 3.0.0
print(sns.__version__)  #> 0.9.0
```