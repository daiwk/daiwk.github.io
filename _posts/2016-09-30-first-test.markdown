---
layout: post
category: "other"
title: "basic usage"
tags: [basic usage]
---

目录

<!-- TOC -->

- [mathjax基本公式语法](#mathjax基本公式语法)

<!-- /TOC -->

## mathjax基本公式语法
[http://blog.csdn.net/ethmery/article/details/50670297](http://blog.csdn.net/ethmery/article/details/50670297)

`\[
P(E)   = {n \choose k} p^k (1-p)^{ n-k}
\tag{Eq-x}
\]`

`\[
\Gamma(n) = (n-1)!\quad\forall n\in\mathbb N
\]`

`\[
x = \dfrac{-b \pm \sqrt{b^2 - 4ac}}{2a}
\]`

In equation \eqref{eq:sample}, we find the value of an
interesting integral:

\begin{equation}
\int_0^\infty \frac{x^3}{e^x-1}\,dx = \frac{\pi^4}{15}
\label{eq:sample}
\end{equation}


| Item      |    Value | Qty  |
| :-------- | --------:| :--: |
| Computer  | 1600 USD |  5   |
| Phone     |   12 USD |  12  |
| Pipe      |    1 USD | 234  |

<pre><code>
@requires_authorization
def somefunc(param1='', param2=0):
    '''A docstring'''
    if param1 > param2: # interesting
        print 'Greater'
    return (param2 - param1 + 1) or None
class SomeClass:
    pass
>>> message = '''interpreter
... prompt'''
</code></pre>

```sequence
Alice->Bob: Hello Bob, how are you?
Note right of Bob: Bob thinks
Bob-->Alice: I am good thanks!
```

 `- [ ] haha`
 `- [x] lalala`