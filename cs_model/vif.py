# -*- coding:utf-8 -*-
__author__ = 'maomaochong'
#http://www.ats.ucla.edu/stat/sas/examples/chp/chp_ch10.htm
from __future__ import division
import numpy as np
import pandas as pd
example = pd.read_csv('by_example_import.csv')
example.dropna(inplace=True)

from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(example)
scaler.transform(example)

X = example.drop(['year', 'import'], axis=1)
#c_matrix = X.corr()
y = example['import']
#w, v = np.linalg.eig(c_matrix)

import pylab as pl
from sklearn import linear_model

###############################################################################
# Compute paths

alphas = [0.000, 0.001, 0.003, 0.005, 0.007, 0.009, 0.010, 0.012, 0.014, 0.016, 0.018,
          0.020, 0.022, 0.024, 0.026, 0.028, 0.030, 0.040, 0.050, 0.060, 0.070, 0.080,
          0.090, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]
clf = linear_model.Ridge(fit_intercept=False)
clf2 = linear_model.Ridge(fit_intercept=False)
coefs = []
vif_list = [[] for x in range(X.shape[1])]
for a in alphas:
    clf.set_params(alpha=a)
    clf.fit(X, y)
    coefs.append(clf.coef_)

    for j, data in enumerate(X.columns):
        cols = [col for col in X.columns if col not in [data]]
        Z = X[cols]
        yy = X.iloc[:,j]
        clf2.set_params(alpha=a)
        clf2.fit(Z, yy)

        r_squared_j = clf2.score(Z, yy)
        vif = 1. / (1. - r_squared_j)
        print r_squared_j
        vif_list[j].append(vif)

pd.DataFrame(vif_list, columns = alphas).T
pd.DataFrame(coefs, index=alphas)

###############################################################################
# Display results

ax = pl.gca()
ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])

ax.plot(alphas, coefs)
pl.vlines(ridge_cv.alpha_, np.min(coefs), np.max(coefs), linestyle='dashdot')
pl.xlabel('alpha')
pl.ylabel('weights')
pl.title('Ridge coefficients as a function of the regularization')
pl.axis('tight')
pl.show()


def vif_ridge(corr_x, pen_factors, is_corr=True):
    """variance inflation factor for Ridge regression

    assumes penalization is on standardized variables
    data should not include a constant

    Parameters
    ----------
    corr_x : array_like
        correlation matrix if is_corr=True or original data if is_corr is False.
    pen_factors : iterable
        iterable of Ridge penalization factors
    is_corr : bool
        Boolean to indicate how corr_x is interpreted, see corr_x

    Returns
    -------
    vif : ndarray
        variance inflation factors for parameters in columns and ridge
        penalization factors in rows

    could be optimized for repeated calculations
    """
    corr_x = np.asarray(corr_x)
    if not is_corr:
        corr = np.corrcoef(corr_x, rowvar=0, bias=True)
    else:
        corr = corr_x

    eye = np.eye(corr.shape[1])
    res = []
    for k in pen_factors:
        minv = np.linalg.inv(corr + k * eye)
        vif = minv.dot(corr).dot(minv)
        res.append(np.diag(vif))
    return np.asarray(res)