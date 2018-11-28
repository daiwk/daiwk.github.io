#encoding=utf8
## 参考https://blog.csdn.net/dengxing1234/article/details/73739836
import xgboost as xgb
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.externals import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.preprocessing.data import OneHotEncoder


def xgboost_lr_train(libsvmFileNameInitial):

    # load样本数据
    X_all, y_all = load_svmlight_file(libsvmFileNameInitial)

    # 训练/测试数据分割
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = 0.3, random_state = 42)

    # 定义xgb模型
    xgboost = xgb.XGBClassifier(nthread=4, learning_rate=0.08,
                            n_estimators=50, max_depth=5, gamma=0, subsample=0.9, colsample_bytree=0.5)
    # 训练xgb学习
    xgboost.fit(X_train, y_train)

    # 预测xgb及AUC评测
    y_pred_test = xgboost.predict_proba(X_test)[:, 1]
    xgb_test_auc = roc_auc_score(y_test, y_pred_test)
    print('xgboost test auc: %.5f' % xgb_test_auc)

    # xgboost编码原有特征
    X_train_leaves = xgboost.apply(X_train)
    X_test_leaves = xgboost.apply(X_test)


    # 合并编码后的训练数据和测试数据
    All_leaves = np.concatenate((X_train_leaves, X_test_leaves), axis=0)
    All_leaves = All_leaves.astype(np.int32)

    # 对所有特征进行ont-hot编码
    xgbenc = OneHotEncoder()
    X_trans = xgbenc.fit_transform(All_leaves)

    (train_rows, cols) = X_train_leaves.shape

    # 定义LR模型
    lr = LogisticRegression()
    # lr对xgboost特征编码后的样本模型训练
    lr.fit(X_trans[:train_rows, :], y_train)
    # 预测及AUC评测
    y_pred_xgblr1 = lr.predict_proba(X_trans[train_rows:, :])[:, 1]
    xgb_lr_auc1 = roc_auc_score(y_test, y_pred_xgblr1)
    print('基于Xgb特征编码后的LR AUC: %.5f' % xgb_lr_auc1)

    # 定义LR模型
    lr = LogisticRegression(n_jobs=-1)
    # 组合特征
    X_train_ext = hstack([X_trans[:train_rows, :], X_train])
    X_test_ext = hstack([X_trans[train_rows:, :], X_test])

    # lr对组合特征的样本模型训练
    lr.fit(X_train_ext, y_train)

    # 预测及AUC评测
    y_pred_xgblr2 = lr.predict_proba(X_test_ext)[:, 1]
    xgb_lr_auc2 = roc_auc_score(y_test, y_pred_xgblr2)
    print('基于组合特征的LR AUC: %.5f' % xgb_lr_auc2)

if __name__ == '__main__':
    xgboost_lr_train("data/sample_libsvm_data.txt")
