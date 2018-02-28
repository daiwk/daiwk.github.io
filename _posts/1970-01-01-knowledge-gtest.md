---
layout: post
category: "knowledge"
title: "gtest"
tags: [gtest, ]
---

目录

<!-- TOC -->

- [gtest](#gtest)
    - [显示所有测试用例](#显示所有测试用例)
    - [只执行部分测试用例](#只执行部分测试用例)

<!-- /TOC -->

## gtest

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

