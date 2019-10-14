---
layout: post
category: "knowledge"
title: "git常用操作"
tags: [git, ]
---

目录

<!-- TOC -->

- [git常用操作](#git常用操作)
    - [如何把fork的源代码的更新merge到自己的来](#如何把fork的源代码的更新merge到自己的来)
    - [如何回滚到某一次commit](#如何回滚到某一次commit)
        - [git reset](#git-reset)
        - [git revert](#git-revert)
    - [如何把一个分支的代码合并到另一个分支](#如何把一个分支的代码合并到另一个分支)
- [如何往github的某一个分支push代码](#如何往github的某一个分支push代码)
- [submodule](#submodule)

<!-- /TOC -->

## git常用操作


### 如何把fork的源代码的更新merge到自己的来

```shell
git remote -v 
git remote add upstream git@github.com:xxx/xxx.git
## 例如： git remote add upstream https://github.com/codertimo/BERT-pytorch.git
git fetch upstream
git merge upstream/master
git push 
```

### 如何回滚到某一次commit

#### git reset

```shell
git reset --hard HEAD^         #回退到上个版本
git reset --hard HEAD~3        #回退到前3次提交之前，以此类推，回退到n次提交之前
git reset --hard commit_id     #退到/进到 指定commit的sha码
```

git revert 和 git reset的区别：

1. git revert是用一次新的commit来回滚之前的commit，git reset是直接删除指定的commit。 
2. 在回滚这一操作上看，效果差不多。但是在日后继续merge以前的老版本时有区别。因为git revert是用一次逆向的commit“中和”之前的提交，因此日后合并老的branch时，导致这部分改变不会再次出现，但是git reset是之间把某些commit在某个branch上删除，因而和老的branch再次merge时，这些被回滚的commit应该还会被引入。 
3. git reset 是把HEAD向后移动了一下，而git revert是HEAD继续前进，只是新的commit的内容和要revert的内容正好相反，能够抵消要被revert的内容。

强行推送到远端

```
git push origin HEAD --force
```

#### git revert

revert操作：
[https://blog.csdn.net/secretx/article/details/51461972](https://blog.csdn.net/secretx/article/details/51461972)

官方文档：

[https://git-scm.com/docs/git-revert](https://git-scm.com/docs/git-revert)

首先，git log看一下

```shell
commit 23bfb9d15d634111111111111 (HEAD -> master, origin/master, origin/HEAD)
Merge: 1f35488 941d4a7
Author: aaa
Date:   Mon Oct 14 19:46:37 2019 +0800

    Merge branch 'branch_xxx' into master

commit 941d4a7939b933333333333333 (branch_xxx)
Author: x11111
Date:   Mon Oct 14 15:28:42 2019 +0800

    change xxx
    
    Change-Id: I934e97d3e08a88888888

commit 669835a2859a24444444444444 (origin/branch_xxx)
Author: x222
Date:   Sat Oct 12 20:14:17 2019 +0800

    change iiiii
    
    Change-Id: I89f1a409776c036222222222

commit 783a64f999999999999999
Merge: 3848c42 1f35488
Author: x3444
Date:   Sat Oct 12 14:57:47 2019 +0800

    Merge remote-tracking branch 'origin/master' into branch_xxx
    
    Change-Id: Ib6af2ff31633333333

commit 1f35488521d47777777777 (origin/branch_iii)
Merge: cf0491a 743b9b1
Author: iimmm
Date:   Sat Oct 12 14:27:12 2019 +0800

    Merge branch 'branch_iii' into master
```

也就是说，整个历史是这样的：

首先branch_iii通过```1f354885```这次commit合到了master，然后在branch_xxx里有```783a64f```、```669835a```、```941d4a7```这么三次commit，最后就把branch_xxx通过```23bfb9d1```这次commit合到了master。

现在想把最后一次commit给revert掉，也就是还原到```1f354885```这次commit的状态。

我们发现，```23bfb9d1```这个commit，写着：```Merge: 1f35488 941d4a7```，这其实包括了两次的commit信息

+ 我们看```941d4a7```开头的这次commit，可见这是合入master前在分支branch_xxx的最后一次commit：

```shell
commit 941d4a7939b933333333333333 (branch_xxx)
Author: x11111
Date:   Mon Oct 14 15:28:42 2019 +0800

    change xxx
    
    Change-Id: I934e97d3e08a88888888
```

+ 我们再看```1f35488```开头的这次commit，可见这是master的上一次合入（从branch_iii）合过来的：

```shell
commit 1f35488521d47777777777 (origin/branch_iii)
Merge: cf0491a 743b9b1
Author: iimmm
Date:   Sat Oct 12 14:27:12 2019 +0800

    Merge branch 'branch_iii' into master
```

我现在是想把最新的这次合入```23bfb9d15d634111111111111```给去掉，也就是我要还原到上次合到master的状态。再看回```Merge: 1f35488 941d4a7```，-m 1表示的是回到```1f35488```这个commit，-m 2表示的是回到```941d4a7```这个commit。我们想要的是把这个分支新加的东西都干掉，所以是回到上次master的commit，也就是```1f35488```，所以我们选择-m 1。

```shell
git revert -m 1 23bfb9d15d634111111111111
```

### 如何把一个分支的代码合并到另一个分支

```shell
git clone xxx
git checkout branch_a
git merge origin/branch_b
# 然后本地处理conflict
git add .
git commit -m "合并分支"
git push origin HEAD:refs/for/branch_a
```

## 如何往github的某一个分支push代码

+ case1

如果已经在github在界面上建好了分支```test-dwk```，那么

```shell
git pull
git branch -a
```

创建本地分支test-dwk(名字必须一致！！)，并且和远程origin/test-dwk分支关联

```shell 
git checkout test-dwk
```

提交
```shell
git push --set-upstream origin test-dwk
```
+ case2

如果没有在github在界面上建好了分支```test-dwk```，那么

```shell
git branch -b test-dwk
```

这样就创建了本地分支test-dwk

提交

```shell
git push --set-upstream origin test-dwk
```

这样，远程也就多了origin/test-dwk这样一个分支

## submodule

[https://www.jianshu.com/p/9000cd49822c](https://www.jianshu.com/p/9000cd49822c)

参考[https://blog.csdn.net/LEON1741/article/details/90259836](https://blog.csdn.net/LEON1741/article/details/90259836)
