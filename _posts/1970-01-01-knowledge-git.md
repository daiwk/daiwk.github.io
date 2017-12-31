---
layout: post
category: "knowledge"
title: "url"
tags: [url, ]
---

目录

<!-- TOC -->

- [git常用操作](#git常用操作)
    - [如何把fork的源代码的更新merge到自己的来](#如何把fork的源代码的更新merge到自己的来)

<!-- /TOC -->

## git常用操作


### 如何把fork的源代码的更新merge到自己的来

```shell
git remote -v 
git remote add upstream git@github.com:xxx/xxx.git
git fetch upstream
git merge upstream/master
git push 
```
