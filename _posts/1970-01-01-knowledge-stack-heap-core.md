---
layout: post
category: "knowledge"
title: "栈、堆、core、gdb等"
tags: [stack, heap, core, 栈, 堆, gdb, ]
---

目录

<!-- TOC -->

- [gdb](#gdb)
- [基础命令](#基础命令)
- [堆与栈](#堆与栈)
- [core](#core)
- [堆、栈导致的core](#堆栈导致的core)
    - [栈空间不足](#栈空间不足)

<!-- /TOC -->

## gdb

首先，我们需要把讨厌的```"Type <return> to continue, or q <return> to quit"```给去掉：

```shell
set pagination off
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
