---
layout: post
category: "knowledge"
title: "eigen"
tags: [eigen, ]
---

目录

<!-- TOC -->

- [入门](#%E5%85%A5%E9%97%A8)

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

