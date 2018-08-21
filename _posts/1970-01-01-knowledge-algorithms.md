---
layout: post
category: "knowledge"
title: "常用算法"
tags: [algorithms, ]
---

目录

<!-- TOC -->

- [蓄水池抽样](#%E8%93%84%E6%B0%B4%E6%B1%A0%E6%8A%BD%E6%A0%B7)

<!-- /TOC -->

### 蓄水池抽样

[https://blog.csdn.net/huagong_adu/article/details/7619665](https://blog.csdn.net/huagong_adu/article/details/7619665)

从一个包含n个对象的列表S中随机选取k个对象，n为一个非常大或者不知道的值。通常情况下，n是一个非常大的值，大到无法一次性把所有列表S中的对象都放到内存中。

如果k=1，我们总是选择第一个对象，以1/2的概率选择第二个，以1/3的概率选择第三个，以此类推，以1/m的概率选择第m个对象。当该过程结束时，每一个对象具有相同的选中概率，即1/n，

证明：第m个对象最终被选中的概率P=选择m的概率*其后面所有对象不被选择的概率

`\[
P=1/m*m/(m+1)*(m+1)/(m+2)*...*(n-1)/n=1/n
\]`

对应蓄水池抽样问题，可以类似的思路解决。

+ 先把读到的前k个对象放入“水库”
+ 对于第k+1个对象开始，以k/(k+1)的概率选择该对象，以k/(k+2)的概率选择第k+2个对象，以此类推，以k/m的概率选择第m个对象（m>k）。
+ 如果m被选中，则随机替换水库中的一个对象。最终每个对象被选中的概率均为k/n，证明如下。

证明：第m个对象被选中的概率=选择m的概率\*（其后元素不被选择的概率+其后元素被选择的概率\*不替换第m个对象的概率），即

`\[
\begin{align*}
P &=\frac{k}{m}*[(\frac{m+1-k}{m+1}+\frac{k}{m+1}*\frac{k-1}{k})*(\frac{m+2-k}{m+2}+\frac{k}{m+2}*\frac{k-1}{k})*...*(\frac{n-k}{n}+\frac{k}{n}*\frac{k-1}{k})] \\ 
 &=k/m*m/n\\
&=k/n
\end{align*}
\]`

```c++
#include <iostream>
#include <cstdlib>
#include <ctime>
 
using namespace std;
 
// generate a random number between i and k,
// both i and k are inclusive.
int randint(int i, int k)
{
    if (i > k)
    {
        int t = i; i = k; k = t; // swap
    }
    int ret = i + rand() % (k - i + 1);
    return ret;
}
 
// take m samples to result from input of n items.
bool reservoir_sampling(const int *input, int n, int *result, int m)
{
    srand(time(NULL));
    if (n < m || input == NULL || result == NULL)
        return false;
    for (int i = 0; i != m; ++i)
        result[i] = input[i];
 
    for (int i = m; i != n; ++i)
    {
        // 对于第i个元素来说，它的概率应该是k/i, 这里的k就是m，也就是从0-i中随机一个数，如果它小于m，那就选中啦
        int j = randint(0, i);
        if (j < m)
            result[j] = input[i];
    }
    return true;
}
 
int main()
{
    const int n = 100;
    const int m = 10;
    int input[n];
    int result[m];
 
    for (int i = 0; i != n; ++i)
        input[i] = i;
    if (reservoir_sampling(input, n, result, m))
        for (int i = 0; i != m; ++i)
            cout << result[i] << " ";
    cout << endl;
    return 0;
}
```