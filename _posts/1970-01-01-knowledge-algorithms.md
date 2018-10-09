---
layout: post
category: "knowledge"
title: "常用算法"
tags: [algorithms, ]
---

目录

<!-- TOC -->

- [蓄水池抽样](#%E8%93%84%E6%B0%B4%E6%B1%A0%E6%8A%BD%E6%A0%B7)
- [轮盘赌](#%E8%BD%AE%E7%9B%98%E8%B5%8C)

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

### 轮盘赌

蚁群算法作为一种启发式算法，在进行路径选择的过程中，当选择下一目标时，通过轮盘赌概率选择的方式完成，这也**保证了每次循环都能随机的命中概率较大的目标。**其算法思路如下：

设P(i)，其中i=1..n，为n个个体被选择的概率，在轮盘上表示为所占扇区的面积百分比，这里显然sum(P)=1。select用来保存n次选择的结果。

[https://blog.csdn.net/zheng_zhiwei/article/details/23209729](https://blog.csdn.net/zheng_zhiwei/article/details/23209729)

[https://blog.csdn.net/u010807846/article/details/51088750](https://blog.csdn.net/u010807846/article/details/51088750)

```c++
#include<math.h>
#include<stdio.h>
#include<stdlib.h>

float generate_random()
{
	//randomly generate number between [0,1]
	float rand_num = 0.0;
	rand_num = (float)rand()/RAND_MAX;
	//printf("%f\n", rand_num);
	return rand_num;
}
 
float *generate_probability_band(float *prob_arr, int length)
{
	//产生概率带
	//传入参数是在main函数中指定的概率分布
	//传出参数就是概率带（积累概率）
	//计算累计概率，保存在sum_array数组中，用malloc写了一下。
	
	printf("一共有%d个概率数据\n",length);
 
	float *sum_array = NULL;
	float *sum_array_tmp = NULL;
	sum_array = (float *)malloc((length+1)*sizeof(float));//多一个是为了保存概率带最左边那个0
	sum_array_tmp = sum_array;
	int i = 0;
	float sum = 0.0;
	sum_array[0] = 0.0;
	for(i;i<length;i++)
	{
		sum = sum + prob_arr[i];
		printf("sum=%f\n",sum);
		sum_array++;//先执行++，目的是把概率带最左边那个0保存在sum_array[0]上
		*sum_array = sum;
		//printf("%x-->%f\n", sum_array,*sum_array);
		
	}
	//sum_array = sum_array_tmp;
	return sum_array_tmp;
	//free(sum_array);
}
 
//判断随机数位于概率带的哪个位置
int *judge_random_location(float *sum_array, int length)
{
	
	int i = 0;
	int j = 0;
	float rand_num=0;
	int *count=NULL;
	count = (int *)malloc((length-1)*sizeof(int)); //count数组用来统计 落在某个概率上的数量
	for(i=0;i<length-1;i++)
	{
		//初始化为0
		count[i] = 0;
	}
 
	for(i=0;i<length;i++)
	{
		printf("sum_array[%d] = %f\n", i, sum_array[i]);
	}
 
	for(j=0;j<1000;j++) //产生1000个随机数
	{
		rand_num = generate_random();
		if(rand_num>0 && rand_num<1)
		{
			for(i=0;i<length;i++)
			{
				if(rand_num>sum_array[i] && rand_num<=sum_array[i+1])
				{
					//对应的概率带 计数器加1
					count[i] = count[i]+1;
				}
 
			}
		}
	}
	
	return count;
}
void main()
{
	int i=0;
 
	//probability array
	float prob_arr[] = {0.1, 0.2, 0.3, 0.4}; //按照概率1:2:3:4
	int length = sizeof(prob_arr)/sizeof(prob_arr[0]);
	float *sum_array = NULL;
	int *count=NULL;
 
	sum_array = generate_probability_band(prob_arr, length);
	count = judge_random_location(sum_array, length+1);
 
	for(i=0;i<length;i++)
	{
		printf("%d:", count[i]);
		if(i == (length-1))
			printf("\b");
	}
	printf("\n");
}

```