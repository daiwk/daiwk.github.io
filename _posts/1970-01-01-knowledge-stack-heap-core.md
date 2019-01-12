---
layout: post
category: "knowledge"
title: "栈、堆、core、gdb等"
tags: [stack, heap, core, 栈, 堆, gdb, ]
---

目录

<!-- TOC -->

- [gdb](#gdb)
- [打印内存](#打印内存)
    - [pretty stl](#pretty-stl)
    - [gogogo](#gogogo)
- [基础命令](#基础命令)
- [堆与栈](#堆与栈)
- [core](#core)
- [堆、栈导致的core](#堆栈导致的core)
    - [栈空间不足](#栈空间不足)
- [其他容易core的点](#其他容易core的点)

<!-- /TOC -->

## gdb

## 打印内存

参考[https://blog.csdn.net/allenlinrui/article/details/5964046](https://blog.csdn.net/allenlinrui/article/details/5964046)

u 表示从当前地址往后请求的字节数，如果不指定的话，GDB**默认是4个bytes**。u参数可以用下面的字符来代替，b表示单字节，h表示双字节，w表示四字 节，g表示八字节。当我们指定了字节长度后，GDB会从指内存定的内存地址开始，读写指定字节，并把其当作一个值取出来。

例如，假设有一个unordered_map a，

```c++
#include <unordered_map>

int main()
{
    std::unordered_map<int, int> a; 
    //a[222] = 333;
    a.insert(std::make_pair(888, 4444)); 
    a.insert(std::make_pair(777888, 44449999));
    return 0;
}
```

那么

```shell
(gdb)  p a._M_h           
$19 = {<std::__detail::_Hashtable_base<int, std::pair<int const, int>, std::__detail::_Select1st, std::equal_to<int>, std::hash<int>, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Hashtable_traits<false, false, true> >> = {<std::__detail::_Hash_code_base<int, std::pair<int const, int>, std::__detail::_Select1st, std::hash<int>, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, false>> = {<std::__detail::_Hashtable_ebo_helper<0, std::__detail::_Select1st, true>> = {<std::__detail::_Select1st> = {<No data fields>}, <No data fields>}, <std::__detail::_Hashtable_ebo_helper<1, std::hash<int>, true>> = {<std::hash<int>> = {<std::__hash_base<unsigned long, int>> = {<No data fields>}, <No data fields>}, <No data fields>}, <std::__detail::_Hashtable_ebo_helper<2, std::__detail::_Mod_range_hashing, true>> = {<std::__detail::_Mod_range_hashing> = {<No data fields>}, <No data fields>}, <No data fields>}, <std::__detail::_Hashtable_ebo_helper<0, std::equal_to<int>, true>> = {<std::equal_to<int>> = {<std::binary_function<int, int, bool>> = {<No data fields>}, <No data fields>}, <No data fields>}, <No data fields>}, <std::__detail::_Map_base<int, std::pair<int const, int>, std::allocator<std::pair<int const, int> >, std::__detail::_Select1st, std::equal_to<int>, std::hash<int>, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Prime_rehash_policy, std::__detail::_Hashtable_traits<false, false, true>, true>> = {<No data fields>}, <std::__detail::_Insert<int, std::pair<int const, int>, std::allocator<std::pair<int const, int> >, std::__detail::_Select1st, std::equal_to<int>, std::hash<int>, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Prime_rehash_policy, std::__detail::_Hashtable_traits<false, false, true>, false, true>> = {<std::__detail::_Insert_base<int, std::pair<int const, int>, std::allocator<std::pair<int const, int> >, std::__detail::_Select1st, std::equal_to<int>, std::hash<int>, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Prime_rehash_policy, std::__detail::_Hashtable_traits<false, false, true> >> = {<No data fields>}, <No data fields>}, <std::__detail::_Rehash_base<int, std::pair<int const, int>, std::allocator<std::pair<int const, int> >, std::__detail::_Select1st, std::equal_to<int>, std::hash<int>, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Prime_rehash_policy, std::__detail::_Hashtable_traits<false, false, true> >> = {<No data fields>}, <std::__detail::_Equality<int, std::pair<int const, int>, std::allocator<std::pair<int const, int> >, std::__detail::_Select1st, std::equal_to<int>, std::hash<int>, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Prime_rehash_policy, std::__detail::_Hashtable_traits<false, false, true>, true>> = {<No data fields>}, _M_buckets = 0x60d140, _M_bucket_count = 11, 
  _M_bbegin = {<std::allocator<std::__detail::_Hash_node<std::pair<int const, int>, false> >> = {<__gnu_cxx::new_allocator<std::__detail::_Hash_node<std::pair<int const, int>, false> >> = {<No data fields>}, <No data fields>}, _M_node = {_M_nxt = 0x60d1c0}}, _M_element_count = 2, _M_rehash_policy = {
    static _S_growth_factor = 2, _M_max_load_factor = 1, _M_next_resize = 11}}
```

可以看到```_M_buckets```的地址是```0x60d140```，所以我们沿着这个地址一直打看看

```shell
(gdb) x/240w 0x60d140
0x60d140:       0       0       4294958720      32767
0x60d150:       0       0       0       0
0x60d160:       0       0       0       0
0x60d170:       0       0       0       0
0x60d180:       6345152 0       0       0
0x60d190:       0       0       33      0
0x60d1a0:       0       0       888     4444
0x60d1b0:       0       0       33      0
0x60d1c0:       6345120 0       777888  44449999
0x60d1d0:       0       0       134705  0
0x60d1e0:       0       0       0       0
0x60d1f0:       0       0       0       0
0x60d200:       0       0       0       0
0x60d210:       0       0       0       0
...
```

### pretty stl

[https://sourceware.org/gdb/wiki/STLSupport](https://sourceware.org/gdb/wiki/STLSupport)

首先的首先！！！确定gdb的版本在7.x以上！！！

首先，例如进入```/home/maude/gdb_printers/```这个文件夹，

```shell
svn co svn://gcc.gnu.org/svn/gcc/trunk/libstdc++-v3/python
```

然后把下面这段贴到```~/.gdbinit``里：

```shell
python
import sys
sys.path.insert(0, '/home/maude/gdb_printers/python')
from libstdcxx.v6.printers import register_libstdcxx_printers
register_libstdcxx_printers (None)
end
```

如果是某个奇葩版本的gcc，例如是4.8.4的[https://www2.cs.duke.edu/csed/cplus/gcc-4.8.4/libstdc++-api-html/a00955_source.html](https://www2.cs.duke.edu/csed/cplus/gcc-4.8.4/libstdc++-api-html/a00955_source.html)，可以修改```libstdcxx/v6/printers.py```：

```python
class StdBaiduHashtableIterator(Iterator):
    def __init__(self, hash):
        self.node = hash['_M_bbegin']['_M_node']['_M_nxt']
        self.node_type = find_type(hash.type, '__node_type').pointer()

    def __iter__(self):
        return self

    def __next__(self):
        if self.node == 0:
            raise StopIteration
        elt = self.node.cast(self.node_type).dereference()
        self.node = elt['_M_nxt']
        #valptr = elt['_M_storage'].address
        valptr = elt.address
        valptr = valptr.cast(elt.type.template_argument(0).pointer())
        #return valptr.dereference()
        return valptr.dereference()
```

调用的时候

```python
        ##data = self.flatten (imap (self.format_one, StdHashtableIterator (self.hashtable())))
        data = self.flatten (imap (self.format_one, StdBaiduHashtableIterator (self.hashtable())))
```

详见[printers_gcc484.py](https://daiwk.github.io/assets/printers_gcc484.py)


当然，更直接的，我们可以用github上的分支！！[https://github.com/gcc-mirror/gcc/tree/gcc-4_8-branch](https://github.com/gcc-mirror/gcc/tree/gcc-4_8-branch)。。。。比较大。。要下好久，建议直接把zip下下来试试。

对应的脚本在[https://github.com/gcc-mirror/gcc/blob/gcc-4_8-branch/libstdc%2B%2B-v3/python/libstdcxx/v6/printers.py](https://github.com/gcc-mirror/gcc/blob/gcc-4_8-branch/libstdc%2B%2B-v3/python/libstdcxx/v6/printers.py)

使用，对于stl的容器，直接

```shell
(gdb) p readlist_news_map 
$1 = std::unordered_map with 10 elements = {[9602626015596390711] = 0x204a8bc08, [9523043020393950406] = 0x1a7b47188, [9331244886375957645] = 0x204a8cd88, 
  [9300120563861322953] = 0x11c951508, [9249702819011901361] = 0x204a8ca08, [9107056655757970751] = 0x1a9e1fb88, [8955599203978976702] = 0x19ab1f488, 
  [8742931355624092612] = 0x204a8ae08, [8551664616403609062] = 0xe0285f88, [10450143932246181310] = 0xe840ae08}
```

如果不想要pretty的格式，直接

```shell
(gdb) p /r readlist_news_map   
$4 = (std::unordered_map<unsigned long, rec::common::RecNewsInfo*, std::hash<unsigned long>, std::equal_to<unsigned long>, std::allocator<std::pair<unsigned long const, rec::common::RecNewsInfo*> > > &) @0x29688320: {<std::__allow_copy_cons<true>> = {<No data fields>}, 
  _M_h = {<std::__detail::_Hashtable_base<unsigned long, std::pair<unsigned long const, rec::common::RecNewsInfo*>, std::__detail::_Select1st, std::equal_to<unsigned long>, std::hash<unsigned long>, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Hashtable_traits<false, false, true> >> = {<std::__detail::_Hash_code_base<unsigned long, std::pair<unsigned long const, rec::common::RecNewsInfo*>, std::__detail::_Select1st, std::hash<unsigned long>, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, false>> = {<std::__detail::_Hashtable_ebo_helper<0, std::__detail::_Select1st, true>> = {<std::__detail::_Select1st> = {<No data fields>}, <No data fields>}, <std::__detail::_Hashtable_ebo_helper<1, std::hash<unsigned long>, true>> = {<std::hash<unsigned long>> = {<std::__hash_base<unsigned long, unsigned long>> = {<No data fields>}, <No data fields>}, <No data fields>}, <std::__detail::_Hashtable_ebo_helper<2, std::__detail::_Mod_range_hashing, true>> = {<std::__detail::_Mod_range_hashing> = {<No data fields>}, <No data fields>}, <No data fields>}, <std::__detail::_Hashtable_ebo_helper<0, std::equal_to<unsigned long>, true>> = {<std::equal_to<unsigned long>> = {<std::binary_function<unsigned long, unsigned long, bool>> = {<No data fields>}, <No data fields>}, <No data fields>}, <No data fields>}, <std::__detail::_Map_base<unsigned long, std::pair<unsigned long const, rec::common::RecNewsInfo*>, std::allocator<std::pair<unsigned long const, rec::common::RecNewsInfo*> >, std::__detail::_Select1st, std::equal_to<unsigned long>, std::hash<unsigned long>, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Prime_rehash_policy, std::__detail::_Hashtable_traits<false, false, true>, true>> = {<No data fields>}, <std::__detail::_Insert<unsigned long, std::pair<unsigned long const, rec::common::RecNewsInfo*>, std::allocator<std::pair<unsigned long const, rec::common::RecNewsInfo*> >, std::__detail::_Select1st, std::equal_to<unsigned long>, std::hash<unsigned long>, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Prime_rehash_policy, std::__detail::_Hashtable_traits<false, false, true>, false, true>> = {<std::__detail::_Insert_base<unsigned long, std::pair<unsigned long const, rec::common::RecNewsInfo*>, std::allocator<std::pair<unsigned long const, rec::common::RecNewsInfo*> >, std::__detail::_Select1st, std::equal_to<unsigned long>, std::hash<unsigned long>, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Prime_rehash_policy, std::__detail::_Hashtable_traits<false, false, true> >> = {<No data fields>}, <No data fields>}, <std::__detail::_Rehash_base<unsigned long, std::pair<unsigned long const, rec::common::RecNewsInfo*>, std::allocator<std::pair<unsigned long const, rec::common::RecNewsInfo*> >, std::__detail::_Select1st, std::equal_to<unsigned long>, std::hash<unsigned long>, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Prime_rehash_policy, std::__detail::_Hashtable_traits<false, false, true> >> = {<No data fields>}, <std::__detail::_Equality<unsigned long, std::pair<unsigned long const, rec::common::RecNewsInfo*>, std::allocator<std::pair<unsigned long const, rec::common::RecNewsInfo*> >, std::__detail::_Select1st, std::equal_to<unsigned long>, std::hash<unsigned long>, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Prime_rehash_policy, std::__detail::_Hashtable_traits<false, false, true>, true>> = {<No data fields>}, _M_buckets = 0x1ce392a80, _M_bucket_count = 47, 
    _M_bbegin = {<std::allocator<std::__detail::_Hash_node<std::pair<unsigned long const, rec::common::RecNewsInfo*>, false> >> = {<__gnu_cxx::new_allocator<std::__detail::_Hash_node<std::pair<unsigned long const, rec::common::RecNewsInfo*>, false> >> = {<No data fields>}, <No data fields>}, _M_node = {
        _M_nxt = 0xe2cabcc0}}, _M_element_count = 10, _M_rehash_policy = {static _S_growth_factor = 2, _M_max_load_factor = 1, _M_next_resize = 47}}}
```

继续探究一下：

```shell
(gdb) p (readlist_news_map._M_h._M_bbegin._M_node._M_nxt) 
$7 = (std::__detail::_Hash_node_base *) 0xe2cabcc0
(gdb) p *(readlist_news_map._M_h._M_bbegin._M_node._M_nxt) 
$8 = {_M_nxt = 0x210605c80}
(gdb) p *(readlist_news_map._M_h._M_bbegin._M_node._M_nxt)._M_nxt
$9 = {_M_nxt = 0xe2caa700}
(gdb) p *(readlist_news_map._M_h._M_bbegin._M_node._M_nxt)._M_nxt._M_nxt
$10 = {_M_nxt = 0xe2caaf60}
(gdb) p *(readlist_news_map._M_h._M_bbegin._M_node._M_nxt)._M_nxt._M_nxt._M_nxt
$11 = {_M_nxt = 0x1f7059200}
(gdb) p *(readlist_news_map._M_h._M_bbegin._M_node._M_nxt)._M_nxt._M_nxt._M_nxt._M_nxt
$12 = {_M_nxt = 0x107a5abc0}
(gdb) p *(readlist_news_map._M_h._M_bbegin._M_node._M_nxt)._M_nxt._M_nxt._M_nxt._M_nxt._M_nxt
$13 = {_M_nxt = 0xe2cab7a0}
(gdb) p *(readlist_news_map._M_h._M_bbegin._M_node._M_nxt)._M_nxt._M_nxt._M_nxt._M_nxt._M_nxt._M_nxt
$14 = {_M_nxt = 0x1b382e0a0}
(gdb) p *(readlist_news_map._M_h._M_bbegin._M_node._M_nxt)._M_nxt._M_nxt._M_nxt._M_nxt._M_nxt._M_nxt._M_nxt
$15 = {_M_nxt = 0x1c1679ce0}
(gdb) p *(readlist_news_map._M_h._M_bbegin._M_node._M_nxt)._M_nxt._M_nxt._M_nxt._M_nxt._M_nxt._M_nxt._M_nxt._M_nxt
$16 = {_M_nxt = 0x1e3d4bd80}
(gdb) p *(readlist_news_map._M_h._M_bbegin._M_node._M_nxt)._M_nxt._M_nxt._M_nxt._M_nxt._M_nxt._M_nxt._M_nxt._M_nxt._M_nxt
$17 = {_M_nxt = 0x0}
```

如果不要中间的星号，那么

```shell
(gdb) p ((readlist_news_map._M_h._M_bbegin._M_node._M_nxt)._M_nxt._M_nxt._M_nxt._M_nxt._M_nxt._M_nxt._M_nxt._M_nxt) 
$33 = (std::__detail::_Hash_node_base *) 0x1c1679ce0
(gdb) p ((readlist_news_map._M_h._M_bbegin._M_node._M_nxt)._M_nxt._M_nxt._M_nxt._M_nxt._M_nxt._M_nxt._M_nxt._M_nxt)._M_nxt
$34 = (std::__detail::_Hash_node_base *) 0x1e3d4bd80
(gdb) p ((readlist_news_map._M_h._M_bbegin._M_node._M_nxt)._M_nxt._M_nxt._M_nxt._M_nxt._M_nxt._M_nxt._M_nxt._M_nxt)._M_nxt._M_nxt
$35 = (std::__detail::_Hash_node_base *) 0x0
```

其实_Hash_node_base就只是一个里面只有一个指针的结构体。。没啥用的样子，我们看看另一个变量

```shell
(gdb) p readlist_news_map._M_h._M_buckets
$46 = (std::_Hashtable<unsigned long, std::pair<unsigned long const, rec::common::RecNewsInfo*>, std::allocator<std::pair<unsigned long const, rec::common::RecNewsInfo*> >, std::__detail::_Select1st, std::equal_to<unsigned long>, std::hash<unsigned long>, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Prime_rehash_policy, std::__detail::_Hashtable_traits<false, false, true> >::__bucket_type *) 0x1ce392a80
```

然后我们参考[https://www2.cs.duke.edu/csed/cplus/gcc-4.8.4/libstdc++-api-html/a00406.html](https://www2.cs.duke.edu/csed/cplus/gcc-4.8.4/libstdc++-api-html/a00406.html)，会发现_M_buckets『好像』是一个数组！

```shell
Each _Hashtable data structure has:

_Bucket[] _M_buckets
_Hash_node_base _M_bbegin
size_type _M_bucket_count
size_type _M_element_count

with _Bucket being _Hash_node* and _Hash_node containing:

_Hash_node* _M_next
Tp _M_value
size_t _M_hash_code if cache_hash_code is true
```

别被骗了。。这是个指针，不是数组[https://www2.cs.duke.edu/csed/cplus/gcc-4.8.4/libstdc++-api-html/a00955_source.html](https://www2.cs.duke.edu/csed/cplus/gcc-4.8.4/libstdc++-api-html/a00955_source.html)：

```c++
__bucket_type*        _M_buckets;
```

```shell
(gdb) p readlist_news_map._M_h._M_buckets[0]
$47 = (std::_Hashtable<unsigned long, std::pair<unsigned long const, rec::common::RecNewsInfo*>, std::allocator<std::pair<unsigned long const, rec::common::RecNewsInfo*> >, std::__detail::_Select1st, std::equal_to<unsigned long>, std::hash<unsigned long>, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Prime_rehash_policy, std::__detail::_Hashtable_traits<false, false, true> >::__bucket_type) 0x0
(gdb) p readlist_news_map._M_h._M_buckets[1]
$48 = (std::_Hashtable<unsigned long, std::pair<unsigned long const, rec::common::RecNewsInfo*>, std::allocator<std::pair<unsigned long const, rec::common::RecNewsInfo*> >, std::__detail::_Select1st, std::equal_to<unsigned long>, std::hash<unsigned long>, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Prime_rehash_policy, std::__detail::_Hashtable_traits<false, false, true> >::__bucket_type) 0x210605c80
(gdb) p readlist_news_map._M_h._M_buckets[2]
$49 = (std::_Hashtable<unsigned long, std::pair<unsigned long const, rec::common::RecNewsInfo*>, std::allocator<std::pair<unsigned long const, rec::common::RecNewsInfo*> >, std::__detail::_Select1st, std::equal_to<unsigned long>, std::hash<unsigned long>, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Prime_rehash_policy, std::__detail::_Hashtable_traits<false, false, true> >::__bucket_type) 0x0
(gdb) p readlist_news_map._M_h._M_buckets[3]
$50 = (std::_Hashtable<unsigned long, std::pair<unsigned long const, rec::common::RecNewsInfo*>, std::allocator<std::pair<unsigned long const, rec::common::RecNewsInfo*> >, std::__detail::_Select1st, std::equal_to<unsigned long>, std::hash<unsigned long>, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Prime_rehash_policy, std::__detail::_Hashtable_traits<false, false, true> >::__bucket_type) 0x0
(gdb) p readlist_news_map._M_h._M_buckets[4]
$51 = (std::_Hashtable<unsigned long, std::pair<unsigned long const, rec::common::RecNewsInfo*>, std::allocator<std::pair<unsigned long const, rec::common::RecNewsInfo*> >, std::__detail::_Select1st, std::equal_to<unsigned long>, std::hash<unsigned long>, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Prime_rehash_policy, std::__detail::_Hashtable_traits<false, false, true> >::__bucket_type) 0x0
(gdb) p readlist_news_map._M_h._M_buckets[5]
$52 = (std::_Hashtable<unsigned long, std::pair<unsigned long const, rec::common::RecNewsInfo*>, std::allocator<std::pair<unsigned long const, rec::common::RecNewsInfo*> >, std::__detail::_Select1st, std::equal_to<unsigned long>, std::hash<unsigned long>, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Prime_rehash_policy, std::__detail::_Hashtable_traits<false, false, true> >::__bucket_type) 0x1b382e0a0
```


再然后看看_Hash_node的定义：[https://www2.cs.duke.edu/csed/cplus/gcc-4.8.4/libstdc++-api-html/a00957_source.html](https://www2.cs.duke.edu/csed/cplus/gcc-4.8.4/libstdc++-api-html/a00957_source.html)

```c++
struct _Hash_node_base
{
  _Hash_node_base* _M_nxt;

  _Hash_node_base() : _M_nxt() { }

  _Hash_node_base(_Hash_node_base* __next) : _M_nxt(__next) { }
};

template<typename _Value>
    struct _Hash_node<_Value, false> : _Hash_node_base
    {
      _Value       _M_v;

      template<typename... _Args>
    _Hash_node(_Args&&... __args)
    : _M_v(std::forward<_Args>(__args)...) { }

      _Hash_node*
      _M_next() const { return static_cast<_Hash_node*>(_M_nxt); }
    };

using __node_base = __detail::_Hash_node_base;
using __bucket_type = __node_base*;
```

......绝望了。。一切都是_Hash_node_base。。。只有个next指针，，玩毛啊

### gogogo

首先，我们需要把讨厌的```"Type <return> to continue, or q <return> to quit"```给去掉：

```shell
set pagination off
```

我们可以发现如果字符串较长，会有"..."，因此可以这么搞：

```shell
(gdb) show print elements
Limit on string chars or array elements to print is 200.
(gdb) set print elements 0
(gdb) show print elements
Limit on string chars or array elements to print is unlimited.
```

另外，可以进行如下设置，这样就可以有缩进地打印了

```shell
set print pretty on 
```

然后一步步打印vector中的元素：

把reslut的第5个元素打出来：

```shell
(gdb) p *(result._M_impl._M_start+4) 
$27 = {<std::__shared_ptr<rec::common::RidTmpInfo, (__gnu_cxx::_Lock_policy)2>> = {_M_ptr = 0x4e653b860, _M_refcount = {_M_pi = 0x4bd3ec80}}, <No data fields>}
```

把它的_M_ptr打出来：

```shell
(gdb) p *(result._M_impl._M_start+4)._M_ptr
$28 = {rid = 6189532051180867495, source_type = 0, cupai_score = 0.528647482, real_score = 0, res_score = 0.581260145, type = 43, mark = 0, category = 0, 
  slotid = 1, predict_result = {static npos = <optimized out>, 
    _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, 
      _M_p = 0x7ff3ac4e93f8 <std::string::_Rep::_S_empty_rep_storage+24> ""}}, predictor_extmsg = {static npos = <optimized out>, 
    _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, 
      _M_p = 0x64f5b5d8 "\032\f\022\002\064\063\032\002\064\063:"}}, news_info = 0x0, video_info = 0x4c090508, click = 0, show = 0, ext_msg_v2 = {
    static npos = <optimized out>, _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, 
      _M_p = 0x1a1ecd648 "EhQZAAAAIK7q4D8gODEAAAAgrurgPxoIEgI0MxoCNDM="}}}
```

把里面的video_info打出来：

```shell
(gdb) p *(result._M_impl._M_start+4)._M_ptr.video_info
$31 = {rid = 13611788894022198871, tag_w = {<std::__allow_copy_cons<true>> = {<No data fields>}, _M_h = {<std::__detail::_Hashtable_base<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float>, std::__detail::_Select1st, std::equal_to<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::hash<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Hashtable_traits<true, false, true> >> = {<std::__detail::_Hash_code_base<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float>, std::__detail::_Select1st, std::hash<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, true>> = {<std::__detail::_Hashtable_ebo_helper<0, std::__detail::_Select1st, true>> = {<std::__detail::_Select1st> = {<No data fields>}, <No data fields>}, <std::__detail::_Hashtable_ebo_helper<1, std::hash<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, true>> = {<std::hash<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >> = {<std::__hash_base<unsigned long, std::basic_string<char, std::char_traits<char>, std::allocator<char> > >> = {<No data fields>}, <No data fields>}, <No data fields>}, <std::__detail::_Hashtable_ebo_helper<2, std::__detail::_Mod_range_hashing, true>> = {<std::__detail::_Mod_range_hashing> = {<No data fields>}, <No data fields>}, <No data fields>}, <std::__detail::_Hashtable_ebo_helper<0, std::equal_to<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, true>> = {<std::equal_to<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >> = {<std::binary_function<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::basic_string<char, std::char_traits<char>, std::allocator<char> >, bool>> = {<No data fields>}, <No data fields>}, <No data fields>}, <No data fields>}, <std::__detail::_Map_base<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float>, std::allocator<std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float> >, std::__detail::_Select1st, std::equal_to<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::hash<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Prime_rehash_policy, std::__detail::_Hashtable_traits<true, false, true>, true>> = {<No data fields>}, <std::__detail::_Insert<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float>, std::allocator<std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float> >, std::__detail::_Select1st, std::equal_to<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::hash<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Prime_rehash_policy, std::__detail::_Hashtable_traits<true, false, true>, false, true>> = {<std::__detail::_Insert_base<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float>, std::allocator<std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float> >, std::__detail::_Select1st, std::equal_to<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::hash<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Prime_rehash_policy, std::__detail::_Hashtable_traits<true, false, true> >> = {<No data fields>}, <No data fields>}, <std::__detail::_Rehash_base<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float>, std::allocator<std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float> >, std::__detail::_Select1st, std::equal_to<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::hash<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Prime_rehash_policy, std::__detail::_Hashtable_traits<true, false, true> >> = {<No data fields>}, <std::__detail::_Equality<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float>, std::allocator<std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float> >, std::__detail::_Select1st, std::equal_to<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::hash<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Prime_rehash_policy, std::__detail::_Hashtable_traits<true, false, true>, true>> = {<No data fields>}, _M_buckets = 0x4c0345a0, _M_bucket_count = 11, _M_bbegin = {<std::allocator<std::__detail::_Hash_node<std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float>, true> >> = {<__gnu_cxx::new_allocator<std::__detail::_Hash_node<std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float>, true> >> = {<No data fields>}, <No data fields>}, _M_node = {_M_nxt = 0x0}}, _M_element_count = 0, _M_rehash_policy = {static _S_growth_factor = 2, _M_max_load_factor = 1, _M_next_resize = 11}}}, manual_tags = {<std::__allow_copy_cons<true>> = {<No data fields>}, _M_h = {<std::__detail::_Hashtable_base<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float>, std::__detail::_Select1st, std::equal_to<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::hash<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Hashtable_traits<true, false, true> >> = {<std::__detail::_Hash_code_base<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float>, std::__detail::_Select1st, std::hash<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, true>> = {<std::__detail::_Hashtable_ebo_helper<0, std::__detail::_Select1st, true>> = {<std::__detail::_Select1st> = {<No data fields>}, <No data fields>}, <std::__detail::_Hashtable_ebo_helper<1, std::hash<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, true>> = {<std::hash<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >> = {<std::__hash_base<unsigned long, std::basic_string<char, std::char_traits<char>, std::allocator<char> > >> = {<No data fields>}, <No data fields>}, <No data fields>}, <std::__detail::_Hashtable_ebo_helper<2, std::__detail::_Mod_range_hashing, true>> = {<std::__detail::_Mod_range_hashing> = {<No data fields>}, <No data fields>}, <No data fields>}, <std::__detail::_Hashtable_ebo_helper<0, std::equal_to<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, true>> = {<std::equal_to<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >> = {<std::binary_function<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::basic_string<char, std::char_traits<char>, std::allocator<char> >, bool>> = {<No data fields>}, <No data fields>}, <No data fields>}, <No data fields>}, <std::__detail::_Map_base<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float>, std::allocator<std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float> >, std::__detail::_Select1st, std::equal_to<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::hash<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Prime_rehash_policy, std::__detail::_Hashtable_traits<true, false, true>, true>> = {<No data fields>}, <std::__detail::_Insert<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float>, std::allocator<std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float> >, std::__detail::_Select1st, std::equal_to<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::hash<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Prime_rehash_policy, std::__detail::_Hashtable_traits<true, false, true>, false, true>> = {<std::__detail::_Insert_base<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float>, std::allocator<std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float> >, std::__detail::_Select1st, std::equal_to<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::hash<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Prime_rehash_policy, std::__detail::_Hashtable_traits<true, false, true> >> = {<No data fields>}, <No data fields>}, <std::__detail::_Rehash_base<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float>, std::allocator<std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float> >, std::__detail::_Select1st, std::equal_to<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::hash<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Prime_rehash_policy, std::__detail::_Hashtable_traits<true, false, true> >> = {<No data fields>}, <std::__detail::_Equality<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float>, std::allocator<std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float> >, std::__detail::_Select1st, std::equal_to<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::hash<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Prime_rehash_policy, std::__detail::_Hashtable_traits<true, false, true>, true>> = {<No data fields>}, _M_buckets = 0x4c034600, _M_bucket_count = 11, _M_bbegin = {<std::allocator<std::__detail::_Hash_node<std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float>, true> >> = {<__gnu_cxx::new_allocator<std::__detail::_Hash_node<std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float>, true> >> = {<No data fields>}, <No data fields>}, _M_node = {_M_nxt = 0x0}}, _M_element_count = 0, _M_rehash_policy = {static _S_growth_factor = 2, _M_max_load_factor = 1, _M_next_resize = 11}}}, videoinfo_pb = {<google::protobuf::Message> = {<google::protobuf::MessageLite> = {_vptr.MessageLite = 0x7ff3a084d5d0 <vtable for rec::doc::VideoInfo+16>}, <No data fields>}, static kRidFieldNumber = 1, static kTitleSignFieldNumber = 2, static kContentSignFieldNumber = 3, static kQualityFieldNumber = 4, static kGenreFieldNumber = 5, static kTagsFieldNumber = 6, static kPublicTimeFieldNumber = 7, static kTabFieldNumber = 8, static kManualTagsFieldNumber = 9, static kNewCateFieldNumber = 10, static kNewSubCateFieldNumber = 11, static kCheckAttributeFieldNumber = 12, static kVideoTypeFieldNumber = 13, static kLongVideoTypeFieldNumber = 14, static kIsCooperateFieldNumber = 15, static kIsNatureFieldNumber = 16, static kCheckpropertyEntityFieldNumber = 17, static kCheckpropertyPeopleFieldNumber = 18, static kCheckpropertySceneFieldNumber = 19, static kDomainFieldNumber = 20, static kLiveTypeFieldNumber = 21, static kUploaderFieldNumber = 22, static kIdlCate1FieldNumber = 23, static kIdlCate2FieldNumber = 24, static kMthidFieldNumber = 25, static kCategoryFieldNumber = 26, static kBigImgFieldNumber = 27, static kDelTagFieldNumber = 28, static kIsMicrovideoFieldNumber = 29, static kAuthorAuthorityScoreV1FieldNumber = 30, _unknown_fields_ = {fields_ = 0x0}, rid_ = 13611788894022198871, title_sign_ = 14762509907472026416, content_sign_ = 14762509907472026416, quality_ = 0, genre_ = 12, tags_ = {<google::protobuf::internal::RepeatedPtrFieldBase> = {static kInitialSize = 4, elements_ = 0x4c0905b8, current_size_ = 0, allocated_size_ = 0, total_size_ = 4, initial_space_ = {0x0, 0x0, 0x0, 0x0}}, <No data fields>}, public_time_ = 1519785790, tab_ = 0x4e65331f8, manual_tags_ = {<google::protobuf::internal::RepeatedPtrFieldBase> = {static kInitialSize = 4, elements_ = 0x4c090600, current_size_ = 1, allocated_size_ = 1, total_size_ = 4, initial_space_ = {0x4e6418990, 0x4e64d0540, 0x0, 0x0}}, <No data fields>}, new_cate_ = 0x4e6533208, new_sub_cate_ = 0x4e6533210, check_attribute_ = {<google::protobuf::internal::RepeatedPtrFieldBase> = {static kInitialSize = 4, elements_ = 0x4c090648, current_size_ = 1, allocated_size_ = 1, total_size_ = 4, initial_space_ = {0x4e6533218, 0x0, 0x0, 0x0}}, <No data fields>}, video_type_ = 0x4e6533220, long_video_type_ = 0x4e6533228, is_cooperate_ = 0, checkproperty_entity_ = {<google::protobuf::internal::RepeatedPtrFieldBase> = {static kInitialSize = 4, elements_ = 0x4c090698, current_size_ = 0, allocated_size_ = 0, total_size_ = 4, initial_space_ = {0x0, 0x0, 0x0, 0x0}}, <No data fields>}, checkproperty_people_ = {<google::protobuf::internal::RepeatedPtrFieldBase> = {static kInitialSize = 4, elements_ = 0x4c0906d0, current_size_ = 0, allocated_size_ = 0, total_size_ = 4, initial_space_ = {0x0, 0x0, 0x0, 0x0}}, <No data fields>}, checkproperty_scene_ = {<google::protobuf::internal::RepeatedPtrFieldBase> = {static kInitialSize = 4, elements_ = 0x4c090708, current_size_ = 0, allocated_size_ = 0, total_size_ = 4, initial_space_ = {0x0, 0x0, 0x0, 0x0}}, <No data fields>}, is_nature_ = 0, idl_cate1_ = 780, domain_ = 0x4e6533230, live_type_ = 0x4e6533238, uploader_ = 0x4e6533240, mthid_ = 0x4e6533248, category_ = 0, idl_cate2_ = 1786, del_tag_ = 0, big_img_ = 0x1de8908 <google::protobuf::internal::kEmptyString>, is_microvideo_ = 0, author_authority_score_v1_ = 0, _cached_size_ = 0, _has_bits_ = {167311071}, static default_instance_ = 0x27a6240}}
```

发现里面的```videoinfo_pb```是我们想要的，于是

```shell
(gdb) p (*(result._M_impl._M_start+4)._M_ptr.video_info).videoinfo_pb
$34 = {<google::protobuf::Message> = {<google::protobuf::MessageLite> = {_vptr.MessageLite = 0x7ff3a084d5d0 <vtable for rec::doc::VideoInfo+16>}, <No data fields>}, static kRidFieldNumber = 1, static kTitleSignFieldNumber = 2, static kContentSignFieldNumber = 3, static kQualityFieldNumber = 4, static kGenreFieldNumber = 5, static kTagsFieldNumber = 6, static kPublicTimeFieldNumber = 7, static kTabFieldNumber = 8, static kManualTagsFieldNumber = 9, static kNewCateFieldNumber = 10, static kNewSubCateFieldNumber = 11, static kCheckAttributeFieldNumber = 12, static kVideoTypeFieldNumber = 13, static kLongVideoTypeFieldNumber = 14, static kIsCooperateFieldNumber = 15, static kIsNatureFieldNumber = 16, static kCheckpropertyEntityFieldNumber = 17, static kCheckpropertyPeopleFieldNumber = 18, static kCheckpropertySceneFieldNumber = 19, static kDomainFieldNumber = 20, static kLiveTypeFieldNumber = 21, static kUploaderFieldNumber = 22, static kIdlCate1FieldNumber = 23, static kIdlCate2FieldNumber = 24, static kMthidFieldNumber = 25, static kCategoryFieldNumber = 26, static kBigImgFieldNumber = 27, static kDelTagFieldNumber = 28, static kIsMicrovideoFieldNumber = 29, static kAuthorAuthorityScoreV1FieldNumber = 30, _unknown_fields_ = {fields_ = 0x0}, rid_ = 13611788894022198871, title_sign_ = 14762509907472026416, content_sign_ = 14762509907472026416, quality_ = 0, genre_ = 12, tags_ = {<google::protobuf::internal::RepeatedPtrFieldBase> = {static kInitialSize = 4, elements_ = 0x4c0905b8, current_size_ = 0, allocated_size_ = 0, total_size_ = 4, initial_space_ = {0x0, 0x0, 0x0, 0x0}}, <No data fields>}, public_time_ = 1519785790, tab_ = 0x4e65331f8, manual_tags_ = {<google::protobuf::internal::RepeatedPtrFieldBase> = {static kInitialSize = 4, elements_ = 0x4c090600, current_size_ = 1, allocated_size_ = 1, total_size_ = 4, initial_space_ = {0x4e6418990, 0x4e64d0540, 0x0, 0x0}}, <No data fields>}, new_cate_ = 0x4e6533208, new_sub_cate_ = 0x4e6533210, check_attribute_ = {<google::protobuf::internal::RepeatedPtrFieldBase> = {static kInitialSize = 4, elements_ = 0x4c090648, current_size_ = 1, allocated_size_ = 1, total_size_ = 4, initial_space_ = {0x4e6533218, 0x0, 0x0, 0x0}}, <No data fields>}, video_type_ = 0x4e6533220, long_video_type_ = 0x4e6533228, is_cooperate_ = 0, checkproperty_entity_ = {<google::protobuf::internal::RepeatedPtrFieldBase> = {static kInitialSize = 4, elements_ = 0x4c090698, current_size_ = 0, allocated_size_ = 0, total_size_ = 4, initial_space_ = {0x0, 0x0, 0x0, 0x0}}, <No data fields>}, checkproperty_people_ = {<google::protobuf::internal::RepeatedPtrFieldBase> = {static kInitialSize = 4, elements_ = 0x4c0906d0, current_size_ = 0, allocated_size_ = 0, total_size_ = 4, initial_space_ = {0x0, 0x0, 0x0, 0x0}}, <No data fields>}, checkproperty_scene_ = {<google::protobuf::internal::RepeatedPtrFieldBase> = {static kInitialSize = 4, elements_ = 0x4c090708, current_size_ = 0, allocated_size_ = 0, total_size_ = 4, initial_space_ = {0x0, 0x0, 0x0, 0x0}}, <No data fields>}, is_nature_ = 0, idl_cate1_ = 780, domain_ = 0x4e6533230, live_type_ = 0x4e6533238, uploader_ = 0x4e6533240, mthid_ = 0x4e6533248, category_ = 0, idl_cate2_ = 1786, del_tag_ = 0, big_img_ = 0x1de8908 <google::protobuf::internal::kEmptyString>, is_microvideo_ = 0, author_authority_score_v1_ = 0, _cached_size_ = 0, _has_bits_ = {167311071}, static default_instance_ = 0x27a6240}
```

再然后，我们想看看new_sub_cate_这个变量：

```shell
(gdb) p (*(result._M_impl._M_start+4)._M_ptr.video_info).videoinfo_pb.new_sub_cate_
$35 = (std::string *) 0x4e6533210
```

再把它的值打出来：

```shell
(gdb) p *((*(result._M_impl._M_start+4)._M_ptr.video_info).videoinfo_pb.new_sub_cate_)
$36 = {static npos = <optimized out>, _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, _M_p = 0x4e64189d8 "\351\243\216\345\260\232\345\244\247\347\211\207"}}
```

然后：

```shell
(gdb) p ((*(result._M_impl._M_start+4)._M_ptr.video_info).videoinfo_pb.new_sub_cate_)._M_dataplus._M_p
$39 = 0x4e64189d8 "\351\243\216\345\260\232\345\244\247\347\211\207"
```

**!!!大boss来了！！**，我们来学习一下下什么叫高科技：(参考[https://www.zhihu.com/question/26902926](https://www.zhihu.com/question/26902926))

```python
#encoding=utf8
import re
import urllib

def ChangeCoding(s):
    ''' 处理中文文件名的编码 '''
#s='"\\346\\226\\260\\345\\273\\272\\346\\226\\207\\344\\273\\266\\345\\244\\271/\\345\\226\\260\\345\\273\\272\\346\\226\\207\\344\\273\\266\\345\\244\\271/\\346\\226\\260\\345\\273\\272\\346\\226\\207\\346\\234\\254\\346\\226\\207\\346\\241\\243.txt"'
#pattern=re.compile(r'^".*?((\\\d\d\d){3,})(/(?P<s>(\\\d\d\d){3,}))*.+"$')
#match=pattern.match(a)

    p=re.compile(r'(?P<s>(\\\d\d\d){3,})')
    for i in p.finditer(s):
        old=i.group('s')
        name=old.split('\\')
        name=['%x' %int(g,8) for g in name if g.isdigit() ]
        name='%'+'%'.join(name)
        CN_name= urllib.unquote(name).decode('utf-8')
        s = s.replace(old,CN_name)
    print s.strip('"')

s = "\347\251\272\345\247\220\350\201\224\347\233\237"
ChangeCoding(s)
```

可以发现结果：

```shell
python /tmp/x.py    
风尚大片
```

看看另一个core，其实类似：

```shell
(gdb) p (*(result._M_impl._M_start+3)._M_ptr.video_info).videoinfo_pb.rid_
$5 = 6842212632
(gdb) p *(result._M_impl._M_start+3)._M_ptr
$8 = {rid = 4279846197472829173, source_type = 0, cupai_score = 0.520694852, real_score = 0, res_score = 0, type = 43, mark = 0, category = 0, slotid = 1, 
  predict_result = {static npos = <optimized out>, 
    _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, 
      _M_p = 0x7fbd7b5823f8 <std::string::_Rep::_S_empty_rep_storage+24> ""}}, predictor_extmsg = {static npos = <optimized out>, 
    _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, 
      _M_p = 0x197e24c18 "\032\f\022\002\064\063\032\002\064\063:"}}, news_info = 0x0, video_info = 0x2cfb188, click = 0, show = 0, ext_msg_v2 = {
    static npos = <optimized out>, _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, 
      _M_p = 0x198255b78 "EhQZAAAAQIip4D8gODEAAABAiKngPxoIEgI0MxoCNDM="}}}
```

发现rid外层是对的，但在内层的videoinfo_pb里却是错的，说明在给videoinfo_pb赋值的地方有问题，进一步地，我们可以看看中层的video_info字段，发现这里的rid也是错的！！发现给这个video_info赋值的代码包在一个```#pragma omp parallel for```里，尝试把这个注释掉…

```shell
(gdb) p (*(result._M_impl._M_start+3)._M_ptr.video_info)
$10 = {rid = 6842219464, tag_w = {<std::__allow_copy_cons<true>> = {<No data fields>}, _M_h = {<std::__detail::_Hashtable_base<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float>, std::__detail::_Select1st, std::equal_to<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::hash<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Hashtable_traits<true, false, true> >> = {<std::__detail::_Hash_code_base<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float>, std::__detail::_Select1st, std::hash<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, true>> = {<std::__detail::_Hashtable_ebo_helper<0, std::__detail::_Select1st, true>> = {<std::__detail::_Select1st> = {<No data fields>}, <No data fields>}, <std::__detail::_Hashtable_ebo_helper<1, std::hash<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, true>> = {<std::hash<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >> = {<std::__hash_base<unsigned long, std::basic_string<char, std::char_traits<char>, std::allocator<char> > >> = {<No data fields>}, <No data fields>}, <No data fields>}, <std::__detail::_Hashtable_ebo_helper<2, std::__detail::_Mod_range_hashing, true>> = {<std::__detail::_Mod_range_hashing> = {<No data fields>}, <No data fields>}, <No data fields>}, <std::__detail::_Hashtable_ebo_helper<0, std::equal_to<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, true>> = {<std::equal_to<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >> = {<std::binary_function<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::basic_string<char, std::char_traits<char>, std::allocator<char> >, bool>> = {<No data fields>}, <No data fields>}, <No data fields>}, <No data fields>}, <std::__detail::_Map_base<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float>, std::allocator<std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float> >, std::__detail::_Select1st, std::equal_to<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::hash<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Prime_rehash_policy, std::__detail::_Hashtable_traits<true, false, true>, true>> = {<No data fields>}, <std::__detail::_Insert<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float>, std::allocator<std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float> >, std::__detail::_Select1st, std::equal_to<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::hash<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Prime_rehash_policy, std::__detail::_Hashtable_traits<true, false, true>, false, true>> = {<std::__detail::_Insert_base<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float>, std::allocator<std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float> >, std::__detail::_Select1st, std::equal_to<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::hash<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Prime_rehash_policy, std::__detail::_Hashtable_traits<true, false, true> >> = {<No data fields>}, <No data fields>}, <std::__detail::_Rehash_base<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float>, std::allocator<std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float> >, std::__detail::_Select1st, std::equal_to<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::hash<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Prime_rehash_policy, std::__detail::_Hashtable_traits<true, false, true> >> = {<No data fields>}, <std::__detail::_Equality<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float>, std::allocator<std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float> >, std::__detail::_Select1st, std::equal_to<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::hash<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Prime_rehash_policy, std::__detail::_Hashtable_traits<true, false, true>, true>> = {<No data fields>}, _M_buckets = 0x197d3fbd0, _M_bucket_count = 6842219480, _M_bbegin = {<std::allocator<std::__detail::_Hash_node<std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float>, true> >> = {<__gnu_cxx::new_allocator<std::__detail::_Hash_node<std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float>, true> >> = {<No data fields>}, <No data fields>}, _M_node = {_M_nxt = 0x197e6fe80}}, _M_element_count = 47244640257, _M_rehash_policy = {static _S_growth_factor = 2, _M_max_load_factor = 1.40129846e-45, _M_next_resize = 6844600064}}}, manual_tags = {<std::__allow_copy_cons<true>> = {<No data fields>}, _M_h = {<std::__detail::_Hashtable_base<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float>, std::__detail::_Select1st, std::equal_to<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::hash<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Hashtable_traits<true, false, true> >> = {<std::__detail::_Hash_code_base<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float>, std::__detail::_Select1st, std::hash<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, true>> = {<std::__detail::_Hashtable_ebo_helper<0, std::__detail::_Select1st, true>> = {<std::__detail::_Select1st> = {<No data fields>}, <No data fields>}, <std::__detail::_Hashtable_ebo_helper<1, std::hash<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, true>> = {<std::hash<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >> = {<std::__hash_base<unsigned long, std::basic_string<char, std::char_traits<char>, std::allocator<char> > >> = {<No data fields>}, <No data fields>}, <No data fields>}, <std::__detail::_Hashtable_ebo_helper<2, std::__detail::_Mod_range_hashing, true>> = {<std::__detail::_Mod_range_hashing> = {<No data fields>}, <No data fields>}, <No data fields>}, <std::__detail::_Hashtable_ebo_helper<0, std::equal_to<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, true>> = {<std::equal_to<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >> = {<std::binary_function<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::basic_string<char, std::char_traits<char>, std::allocator<char> >, bool>> = {<No data fields>}, <No data fields>}, <No data fields>}, <No data fields>}, <std::__detail::_Map_base<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float>, std::allocator<std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float> >, std::__detail::_Select1st, std::equal_to<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::hash<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Prime_rehash_policy, std::__detail::_Hashtable_traits<true, false, true>, true>> = {<No data fields>}, <std::__detail::_Insert<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float>, std::allocator<std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float> >, std::__detail::_Select1st, std::equal_to<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::hash<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Prime_rehash_policy, std::__detail::_Hashtable_traits<true, false, true>, false, true>> = {<std::__detail::_Insert_base<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float>, std::allocator<std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float> >, std::__detail::_Select1st, std::equal_to<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::hash<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Prime_rehash_policy, std::__detail::_Hashtable_traits<true, false, true> >> = {<No data fields>}, <No data fields>}, <std::__detail::_Rehash_base<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float>, std::allocator<std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float> >, std::__detail::_Select1st, std::equal_to<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::hash<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Prime_rehash_policy, std::__detail::_Hashtable_traits<true, false, true> >> = {<No data fields>}, <std::__detail::_Equality<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float>, std::allocator<std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float> >, std::__detail::_Select1st, std::equal_to<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::hash<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Prime_rehash_policy, std::__detail::_Hashtable_traits<true, false, true>, true>> = {<No data fields>}, _M_buckets = 0x0, _M_bucket_count = 6844599824, _M_bbegin = {<std::allocator<std::__detail::_Hash_node<std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float>, true> >> = {<__gnu_cxx::new_allocator<std::__detail::_Hash_node<std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, float>, true> >> = {<No data fields>}, <No data fields>}, _M_node = {_M_nxt = 0x0}}, _M_element_count = 0, _M_rehash_policy = {static _S_growth_factor = 2, _M_max_load_factor = 1.65121836e-37, _M_next_resize = 0}}}, videoinfo_pb = {<google::protobuf::Message> = {<google::protobuf::MessageLite> = {_vptr.MessageLite = 0x10ae850 <vtable for google::protobuf::MessageLite+16>}, <No data fields>}, static kRidFieldNumber = 1, static kTitleSignFieldNumber = 2, static kContentSignFieldNumber = 3, static kQualityFieldNumber = 4, static kGenreFieldNumber = 5, static kTagsFieldNumber = 6, static kPublicTimeFieldNumber = 7, static kTabFieldNumber = 8, static kManualTagsFieldNumber = 9, static kNewCateFieldNumber = 10, static kNewSubCateFieldNumber = 11, static kCheckAttributeFieldNumber = 12, static kVideoTypeFieldNumber = 13, static kLongVideoTypeFieldNumber = 14, static kIsCooperateFieldNumber = 15, static kIsNatureFieldNumber = 16, static kCheckpropertyEntityFieldNumber = 17, static kCheckpropertyPeopleFieldNumber = 18, static kCheckpropertySceneFieldNumber = 19, static kDomainFieldNumber = 20, static kLiveTypeFieldNumber = 21, static kUploaderFieldNumber = 22, static kIdlCate1FieldNumber = 23, static kIdlCate2FieldNumber = 24, static kMthidFieldNumber = 25, static kCategoryFieldNumber = 26, static kBigImgFieldNumber = 27, static kDelTagFieldNumber = 28, static kIsMicrovideoFieldNumber = 29, static kAuthorAuthorityScoreV1FieldNumber = 30, _unknown_fields_ = {fields_ = 0x197d3e120}, rid_ = 6842212632, title_sign_ = 6842212640, content_sign_ = 6842212648, quality_ = -1.49276465e-24, genre_ = 1, tags_ = {<google::protobuf::internal::RepeatedPtrFieldBase> = {static kInitialSize = 4, elements_ = 0xb00000002, current_size_ = 1, allocated_size_ = 0, total_size_ = -1745334528, initial_space_ = {0x0, 0x197f84e10, 0x0, 0x0}}, <No data fields>}, public_time_ = 39895184, tab_ = 0x426f4300, manual_tags_ = {<google::protobuf::internal::RepeatedPtrFieldBase> = {static kInitialSize = 4, elements_ = 0x2cfb280, current_size_ = -1747721928, allocated_size_ = 1, total_size_ = -1747721936, initial_space_ = {0x197d3e138, 0x197d3e140, 0x197e6fe80, 0xb00000003}}, <No data fields>}, new_cate_ = 0x1, new_sub_cate_ = 0x197f84f00, check_attribute_ = {<google::protobuf::internal::RepeatedPtrFieldBase> = {static kInitialSize = 4, elements_ = 0x0, current_size_ = -1745334768, allocated_size_ = 1, total_size_ = 0, initial_space_ = {0x0, 0x260c090, 0x0, 0x0}}, <No data fields>}, video_type_ = 0x197d3e150, long_video_type_ = 0x197d3e148, is_cooperate_ = 6842212688, checkproperty_entity_ = {<google::protobuf::internal::RepeatedPtrFieldBase> = {static kInitialSize = 4, elements_ = 0x197d3e158, current_size_ = -1746469248, allocated_size_ = 1, total_size_ = 4, initial_space_ = {0x1, 0x197f84f00, 0x0, 0x197f84e10}}, <No data fields>}, checkproperty_people_ = {<google::protobuf::internal::RepeatedPtrFieldBase> = {static kInitialSize = 4, elements_ = 0x0, current_size_ = 0, allocated_size_ = 0, total_size_ = 39895184, initial_space_ = {0x0, 0x0, 0x197d3e168, 0x197d3e160}}, <No data fields>}, checkproperty_scene_ = {<google::protobuf::internal::RepeatedPtrFieldBase> = {static kInitialSize = 4, elements_ = 0x197d3e168, current_size_ = -1747721872, allocated_size_ = 1, total_size_ = -1746469248, initial_space_ = {0xb00000005, 0x3, 0x197f84f00, 0x0}}, <No data fields>}, is_nature_ = -1745334648, idl_cate1_ = 1, domain_ = 0x0, live_type_ = 0x0, uploader_ = 0x260c090, mthid_ = 0x426f4300, category_ = 0, idl_cate2_ = 4279, del_tag_ = 0, big_img_ = 0x1de8908 <google::protobuf::internal::kEmptyString>, is_microvideo_ = 0, author_authority_score_v1_ = 0, _cached_size_ = 0, _has_bits_ = {167311071}, static default_instance_ = 0x27a6240}}
```

然而问题并不在这里。。。后来仔细review代码发现：

```c++
int func_aaa(TmpResultBuffer& tmp_res) {

TmpResultBuffer mid_tmp_res;
mid_tmp_res.init(200);
if (!func_xxx(context, rid_sim_vec_trunc, mid_tmp_res)) {
    continue;
}

// ...

uint64_t adress = mid_tmp_res.get_doc_info(it->rid);
const rec::common::RecVideoInfo* videoinfo_ptr = (const rec::common::RecVideoInfo*) adress;
RidTmpInfoPtr rid_tmp_info = mid_tmp_res.data[tmp_res_len];
rid_tmp_info->video_info = videoinfo_ptr;
content_v.push_back(std::move(rid_tmp_info));

//...

size_t res_len = tmp_res.tmp_result_len;
for (auto& content_list : all_content) {
    for (auto& content : content_list) {
        if (tmp_set.find(content->video_info->rid) != tmp_set.end()) {
            continue;
        }
        tmp_set.insert(content->video_info->rid);
        tmp_res.data[res_len] = std::move(content);
        ++res_len;
    }
}
tmp_res.tmp_result_len = res_len;

}
```

也就是说，临时变量```mid_tmp_res```里的一个地址```adress```被赋值给了```rid_tmp_info```这个变量的```video_info```这个成员！！但```mid_tmp_res```这个变量在函数执行完后就被回收了，所以后面还想要用```tmp_res```里的成员变量的```video_info```的时候，会发现这个地址取出来的值是不对的！！。。。解决方法呢，就是把这个```mid_tmp_res```的生命周期变得和```tmp_res```一样长了，看框架的设计机制咯。。

## 基础命令

```shell
ulimit -a
core file size          (blocks, -c) unlimited
data seg size           (kbytes, -d) unlimited
file size               (blocks, -f) unlimited
pending signals                 (-i) 1031511
max locked memory       (kbytes, -l) 64
max memory size         (kbytes, -m) unlimited
open files                      (-n) 10240
pipe size            (512 bytes, -p) 8
POSIX message queues     (bytes, -q) 819200
stack size              (kbytes, -s) 10240
cpu time               (seconds, -t) unlimited
max user processes              (-u) 1031511
virtual memory          (kbytes, -v) unlimited
file locks                      (-x) unlimited
```

可见，stack size即栈空间的大小是10240KB，也就是10MB。可用ulimit -s可以只看栈空间大小。

## 堆与栈

## core

[https://blog.csdn.net/caspiansea/article/details/24450377](https://blog.csdn.net/caspiansea/article/details/24450377)

已经core了的文件好像查不了mappings，用gdb启动的可以看：

```shell
info proc  mappings
```

前面会有：

```shell
(gdb) info proc  mappings   
process 544  
Mapped address spaces:  
  
    Start Addr   End Addr       Size     Offset objfile  
        0x8000     0x9000     0x1000        0x0 /mnt/test_class  
       0x10000    0x11000     0x1000        0x0 /mnt/test_class  
       0x11000    0x32000    0x21000        0x0 [heap]  
    0xb6d39000 0xb6e64000   0x12b000        0x0 /lib/libc-2.19.so  
    0xb6e64000 0xb6e6c000     0x8000   0x12b000 /lib/libc-2.19.so  
```

说明0x11000-0x32000这总共0x21000的大小是堆空间

最后面会有：

```shell
      0x7ffff7ffb000     0x7ffff7ffc000     0x1000        0x0 [vdso]
      0x7ffff7ffc000     0x7ffff7ffd000     0x1000    0x20000 /home/opt/gcc-4.8.2.bpkg-r4/gcc-4.8.2.bpkg-r4/lib64/ld-2.18.so
      0x7ffff7ffd000     0x7ffff7ffe000     0x1000    0x21000 /home/opt/gcc-4.8.2.bpkg-r4/gcc-4.8.2.bpkg-r4/lib64/ld-2.18.so
      0x7ffff7ffe000     0x7ffff7fff000     0x1000        0x0 
      0x7ffffff73000     0x7ffffffff000    0x8c000        0x0 [stack]
  0xffffffffff600000 0xffffffffff601000     0x1000        0x0 [vsyscall]
```

说明0x7ffffff73000-0x7ffffffff000这总共0x8c000=789999=789k=0.8MB的大小是栈空间？？好像不太对呢。。。

查看当前frame：

```shell
info frame
Stack level 1, frame at 0x7f7ed3284310:
 rip = 0x7f7ed6da1f50 in nerl::NerlPlus::tagging (baidu/xxxx/src/dddd.cpp:599); saved rip = 0x7f7ed6d9964e
 called by frame at 0x7f7ed3284360, caller of frame at 0x7f7ed32841e0
 source language c++.
 Arglist at 0x7f7ed32841d8, args: this=0x2fd6950, iTokens=0x7f7caae7f010, iTokensCount=1, iNerlBuff=0x7f7c9f706710, tmpTags=..., oTags=..., 
    flags=nerl::DEFAULT_FLAGS
 Locals at 0x7f7ed32841d8, Previous frame's sp is 0x7f7ed3284310
 Saved registers:
  rbx at 0x7f7ed32842d8, rbp at 0x7f7ed32842e0, r12 at 0x7f7ed32842e8, r13 at 0x7f7ed32842f0, r14 at 0x7f7ed32842f8, r15 at 0x7f7ed3284300,
  rip at 0x7f7ed3284308
```

## 堆、栈导致的core

### 栈空间不足

参考：[https://blog.csdn.net/u011866460/article/details/42525171](https://blog.csdn.net/u011866460/article/details/42525171)

例如，程序中有两个大小为`\(2048*2048\)`的char数组，算下来，一个char是一个字节，两个`\(2048*2048\)`的数组便是`\(2*2048*2048=8388608=8*1024*1024=8MB\)`的空间。所以，如果这个时候还有别的栈上的变量，而栈空间如果 只有8MB，那么，就会core!!!

linux限制了栈空间大小，自己定义的变量都是在栈空间上分配的，子函数在调用时才会装入栈中，当定义的变量过大则会超出栈空间，从而段错误。所以，尽可能使用堆空间，比如用new malloc vector等


## 其他容易core的点

+ 类定义时会给这个指针默认分配一个地址，不是0=NULL=nullptr，如果没有new一个这个类型的，就直接用，会core

另外，了解一下nullptr和NULL等的区别：[https://www.cnblogs.com/DswCnblog/p/5629073.html](https://www.cnblogs.com/DswCnblog/p/5629073.html)

```c++
#include <iostream>

using namespace std;

class A {
    public:
        int* a;
        A * b;
};

int main()
{
    A xx;
    if (xx.a == 0) {
        cout << xx.a << "is 0" << endl; //<< nullptr << xx.a << endl;
    }
    if (xx.a == NULL) {
        cout << xx.a << "is NULL" << endl; //<< nullptr << xx.a << endl;
    }
    if (nullptr == NULL) {
        cout << "nullptr==NULL" << endl; //<< nullptr << xx.a << endl;
    }
    if (nullptr == 0) {
        cout << "nullptr==0" << endl; //<< nullptr << xx.a << endl;
    }

    if (xx.a != nullptr) {
        cout << xx.a << "before not nullptr" << endl; //<< nullptr << xx.a << endl;
        *(xx.a) = 4; // 这里ok
        *((xx.b)->a)=444; // 这里会core..
        cout << xx.a << "after not nullptr" << endl; //<< nullptr << xx.a << endl;
    } else {
        cout << xx.a << "nullptr" << endl; //<< nullptr << xx.a << endl;
    }
    return 0;
}
```
