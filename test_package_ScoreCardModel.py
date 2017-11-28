from sklearn import datasets
import pandas as pd
from ScoreCardModel.binning.discretization import Discretization
from ScoreCardModel.weight_of_evidence import WeightOfEvidence
from ScoreCardModel.models.logistic_regression_model import LogisticRegressionModel
from ScoreCardModel.score_card import ScoreCardModel


class MyLR(LogisticRegressionModel):
    def predict(self, x):
        x = self.pre_trade(x)
        return self._predict_proba(x)

    def pre_trade(self, x):
        import numpy as np
        result = []
        for i, v in x.items():
            t = self.ds[i].transform([v])[0]
            r = self.woes[i].transform([t])[0]
            result.append(r)
        return np.array(result)

    def _pre_trade_batch_row(self, row, Y, bins):
        d = Discretization(bins)
        d_row = d.transform(row)
        woe = WeightOfEvidence()
        woe.fit(d_row, Y)
        return d, woe, woe.transform(d_row)

    def pre_trade_batch(self, X, Y):
        self.ds = {}
        self.woes = {}
        self.table = {}
        self.ds["sepal length (cm)"], self.woes["sepal length (cm)"], self.table[
            "sepal length (cm)"] = self._pre_trade_batch_row(
            X["sepal length (cm)"], Y, [0, 2, 5, 8])
        self.ds['sepal width (cm)'], self.woes['sepal width (cm)'], self.table[
            'sepal width (cm)'] = self._pre_trade_batch_row(
            X['sepal width (cm)'], Y, [0, 2, 2.5, 3, 3.5, 5])
        self.ds['petal length (cm)'], self.woes['petal length (cm)'], self.table[
            'petal length (cm)'] = self._pre_trade_batch_row(
            X['petal length (cm)'], Y, [0, 1, 2, 3, 4, 5, 7])
        self.ds['petal width (cm)'], self.woes['petal width (cm)'], self.table[
            'petal width (cm)'] = self._pre_trade_batch_row(
            X['petal width (cm)'], Y, [0, 1, 2, 3])
        return pd.DataFrame(self.table)


iris = datasets.load_iris()
y = iris.target
z = (y == 0)
l = pd.DataFrame(iris.data, columns=iris.feature_names)
lr = MyLR()
lr.train(l, z)
lr.predict(l.loc[0].to_dict())
sc = ScoreCardModel(lr)
sc.predict(l.loc[0].to_dict())
sc_str = sc.dumps()
sc_l = ScoreCardModel.loads(sc_str)
sc_l.predict(l.loc[0].to_dict())
