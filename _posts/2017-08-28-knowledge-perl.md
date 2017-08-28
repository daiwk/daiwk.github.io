---
layout: post
category: "knowledge"
title: "perl"
tags: [perl]
---

## 单/双引号
双引号可以正常解析一些转义字符与变量，而单引号无法解析会原样输出。

```perl
$a = 10;
print "a = $a\n"; # a = 10
print 'a = $a\n'; # a = $a\n
```

## Here文档

Here文档又称作heredoc、hereis、here-字串或here-脚本，是一种在命令行shell（如sh、csh、ksh、bash、PowerShell和zsh）和程序语言（像Perl、PHP、Python和Ruby）里定义一个字串的方法。

+ 1.必须后接分号，否则编译通不过。
+ 2.END可以用任意其它字符代替，只需保证结束标识与开始标识一致。
+ 3.结束标识必须顶格独自占一行(即必须从行首开始，前后不能衔接任何空白和字符)。
+ 4.开始标识可以不带引号号或带单双引号，不带引号与带双引号效果一致，解释内嵌的变量和转义符号，带单引号则不解释内嵌的变量和转义符号。
+ 5.当内容需要内嵌引号（单引号或双引号）时，不需要加转义符，本身对单双引号转义，此处相当与q和qq的用法。

```perl
#!/usr/bin/perl
 
$a = 10;
$var = <<"EOF";
这是一个 Here 文档实例，使用双引号。
可以在这输如字符串和变量。
例如：a = $a
EOF
print "$var\n";
 
$var = <<'EOF';
这是一个 Here 文档实例，使用单引号。
例如：a = $a
EOF
print "$var\n";
# 输出
# 这是一个 Here 文档实例，使用双引号。
# 可以在这输如字符串和变量。
# 例如：a = 10
# 
# 这是一个 Here 文档实例，使用单引号。
# 例如：a = $a
```

## 数据类型

+ 标量
```perl
$myfirst=123;      #数字123　
$mysecond="123";   #字符串123　
```
+ 数组
```perl
@arr=(1,2,3)
```
+ 哈希
```perl
%h=('a'=>1,'b'=>2); 
```
各种转义字符的意思详见[http://www.runoob.com/perl/perl-data-types.html](http://www.runoob.com/perl/perl-data-types.html)

```perl
#!/usr/bin/perl
 
%data = ('google', 45, 'runoob', 30, 'taobao', 40);
 
print "\$data{'google'} = $data{'google'}\n";
print "\$data{'runoob'} = $data{'runoob'}\n";
print "\$data{'taobao'} = $data{'taobao'}\n";
```

