---
layout: post
category: "knowledge"
title: "eigen"
tags: [eigen, ]
---

目录

<!-- TOC -->

- [入门](#%e5%85%a5%e9%97%a8)
- [基本操作](#%e5%9f%ba%e6%9c%ac%e6%93%8d%e4%bd%9c)
- [eigen+mkl加速](#eigenmkl%e5%8a%a0%e9%80%9f)
- [常用函数](#%e5%b8%b8%e7%94%a8%e5%87%bd%e6%95%b0)
  - [block](#block)

<!-- /TOC -->

## 入门

安装只需要把[http://eigen.tuxfamily.org/index.php?title=Main_Page#Download](http://eigen.tuxfamily.org/index.php?title=Main_Page#Download)的包下载下来，然后解压，把**Eigen**目录拿出来就行了，不用编译成so什么的！！

比如是这个目录```/home/work/my_lib/Eigen```

有github的mirror：[https://github.com/eigenteam/eigen-git-mirror](https://github.com/eigenteam/eigen-git-mirror)

```c++
#include <iostream>
#include <Eigen/Dense>
using Eigen::MatrixXd;
int main()
{
  MatrixXd m(2,2);
  m(0,0) = 3;
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = m(1,0) + m(0,1);
  std::cout << m << std::endl;
}
```

编译：

```shell
g++ -I /home/work/my_lib/ my_program.cpp -o my_program
```

运行结果：

```shell
  3  -1
2.5 1.5
```

## 基本操作

参考[https://blog.csdn.net/y363703390/article/details/78407346](https://blog.csdn.net/y363703390/article/details/78407346)

注意，eigen的vector默认是个列向量：

```c++
Eigen::Vector3d v1(1,2,3);
td::cout << "here is v1:" << v1 << std::endl;
// 输出：
// here is v1:1
// 2
// 3
```

向量点积是dot，而cross product只对size为3的vector生效

VectorXf和MatrixXf可以搞出指定size的向量和矩阵：

## eigen+mkl加速

首先装一下mkl

然后在编译的时候加上(我也不知道后面这坨哪个比较有用。。)：

```c++
-DEIGEN_USE_MKL_ALL -msse -msse2 -msse3 -lmkl_core -lmkl_gf_lp64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_def -lmkl_mc3
```

链接时加上

```c++
-ldl
```

参考的是[http://eigen.tuxfamily.org/dox-3.2/TopicUsingIntelMKL.html](http://eigen.tuxfamily.org/dox-3.2/TopicUsingIntelMKL.html)

其中的```-DEIGEN_USE_MKL_ALL```相当于在cpp文件一开头定义如下宏，然后再include相关eigen库：

```c++
#define EIGEN_USE_MKL_ALL
```

## 常用函数

### block

截取从startRow开始blockRows行，以及从startCol开始blockCols列的小矩阵出来。

```c++
inline Block<Derived> block(Index startRow, Index startCol, Index blockRows, Index blockCols)
{
  return Block<Derived>(derived(), startRow, startCol, blockRows, blockCols);
}
/** This is the const version of block(Index,Index,Index,Index). */
inline const Block<const Derived> block(Index startRow, Index startCol, Index blockRows, Index blockCols) const
{
  return Block<const Derived>(derived(), startRow, startCol, blockRows, blockCols);
}
```
