# Copyright 2019 The grace_t Authors. All Rights Reserved.
#  
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#  
# http://www.apache.org/licenses/LICENSE-2.0
#  
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

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
