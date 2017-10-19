# ScoreCard
用Python实现评分卡模型，将会包含如下模块：<br>
1/配置文件<br>
2/数据加载：csv,mysql<br>
3/数据变换：数据类型变换<br>
4/分箱：连续变量和离散变量分箱处理<br>
5/WOE转换：根据分箱后的变量计算某个变量在某个取值上对异质性消除的重要性<br>
6/IV值：根据WOE值计算某个变量对异质性消除的重要性<br>
7/LogisticRegression:根据转换后的WOE值训练逻辑回归模型，并预测概率值<br>
8/模型评估：ROC/AUC/KS<br>
