#!/usr/bin/env bash

#计算auc,输入分别为预测值（可以乘以一个倍数之后转化为整数），该相同预测值的样本个数，该相同预测值的正样本个数
sort -t $'\t' -k 1,1n | awk -F"\t" 'BEGIN{
    OFS="\t";
    now_q="";
    begin_rank=1;
    now_pos_num=0;
    now_neg_num=0;
    total_pos_rank=0;
    total_pos_num=0;
    total_neg_num=0;
}function clear(){
    begin_rank += now_pos_num + now_neg_num;
    now_pos_num=0;
    now_neg_num=0;
}function update(){
    now_pos_num += pos_num;
    now_neg_num += neg_num;
}function output(){
    n = now_pos_num + now_neg_num;
    avg_rank = begin_rank + (n-1)/2;
    tmp_all_pos_rank = avg_rank * now_pos_num;
    total_pos_rank += tmp_all_pos_rank;
    total_pos_num += now_pos_num;
    total_neg_num += now_neg_num;
}{
    q=$1;
    show=$2;
    clk=$3;
    pos_num=clk;
    neg_num=show-clk;
    if(now_q!=q){
        if(now_q!=""){
            output();
            clear();
        }
        now_q=q;
    }
    update();

}END{
    output();
    auc=0;
    m=total_pos_num;
       n=total_neg_num;
    if(m>0 && n>0){
        auc = (total_pos_rank-m*(m+1)/2) / (m*n);
    }
    print auc;
}'