---
layout: post
category: "other"
title: "时间序列预测算法"
tags: [时间序列预测算法, ]
---

目录

<!-- TOC -->

- [holt winters](#holt-winters)
  - [一次指数平滑](#%E4%B8%80%E6%AC%A1%E6%8C%87%E6%95%B0%E5%B9%B3%E6%BB%91)
  - [二次指数平滑](#%E4%BA%8C%E6%AC%A1%E6%8C%87%E6%95%B0%E5%B9%B3%E6%BB%91)
  - [三次指数平滑](#%E4%B8%89%E6%AC%A1%E6%8C%87%E6%95%B0%E5%B9%B3%E6%BB%91)

<!-- /TOC -->

参考[https://www.zhihu.com/question/21229371](https://www.zhihu.com/question/21229371)

参考[https://www.analyticsvidhya.com/blog/2018/02/time-series-forecasting-methods/](https://www.analyticsvidhya.com/blog/2018/02/time-series-forecasting-methods/)

比较流行的还有holt winters、arima等，在statsmodels这个lib里就有：

参考[https://www.jianshu.com/p/2c607fe926f0](https://www.jianshu.com/p/2c607fe926f0)

## holt winters

### 一次指数平滑

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

x1 = np.linspace(0, 1, 100)
y1 = pd.Series(np.multiply(x1, (x1 - 0.5)) + np.random.randn(100))
ets1 = SimpleExpSmoothing(y1)
r1 = ets1.fit()
pred1 = r1.predict(start=len(y1), end=len(y1) + len(y1)//2)

pd.DataFrame({
    'origin': y1,
    'fitted': r1.fittedvalues,
    'pred': pred1
}).plot(legend=True)
```

### 二次指数平滑

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import Holt

x2 = np.linspace(0, 99, 100)
y2 = pd.Series(0.1 * x2 + 2 * np.random.randn(100))
ets2 = Holt(y2)
r2 = ets2.fit()
pred2 = r2.predict(start=len(y2), end=len(y2) + len(y2)//2)

pd.DataFrame({
    'origin': y2,
    'fitted': r2.fittedvalues,
    'pred': pred2
}).plot(legend=True)
```

### 三次指数平滑

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing

x3 = np.linspace(0, 4 * np.pi, 100)
y3 = pd.Series(20 + 0.1 * np.multiply(x3, x3) + 8 * np.cos(2 * x3) + 2 * np.random.randn(100))
ets3 = ExponentialSmoothing(y3, trend='add', seasonal='add', seasonal_periods=25)
r3 = ets3.fit()
pred3 = r3.predict(start=len(y3), end=len(y3) + len(y3)//2)

pd.DataFrame({
    'origin': y3,
    'fitted': r3.fittedvalues,
    'pred': pred3
}).plot(legend=True)
```
