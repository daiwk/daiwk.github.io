---
layout: post
category: "knowledge"
title: "单元测试工具"
tags: [gtest, BllseyeCoverage, ]
---

目录

<!-- TOC -->

- [gtest](#gtest)
    - [显示所有测试用例](#显示所有测试用例)
    - [只执行部分测试用例](#只执行部分测试用例)
- [BullseyeCoverage](#bullseyecoverage)

<!-- /TOC -->

## gtest

简介：[Google C++单元测试框架---Gtest框架简介（译文）](https://www.cnblogs.com/jycboy/p/6057677.html)

### 显示所有测试用例

```
./xxx --gtest_list_tests
```

### 只执行部分测试用例

例如上面的输出是

```shell
TestNodeA.
  test_a1
  test_a2
TestNodeB.
  test_b1
  test_b2
  test_b3
  test_b4
```

那么，可以

```shell
./xxx --gtest_filter=TestNodeA.*
./xxx --gtest_filter=TestNodeA.test*
./xxx --gtest_filter=TestNodeA.test_a1
```

## BullseyeCoverage

官网的下载页面：[https://www.bullseye.com/cgi-bin/download.sh](https://www.bullseye.com/cgi-bin/download.sh)

简介：[代码覆盖率工具BullseyeCoverage研究](http://blog.csdn.net/billbliss/article/details/43971629)

