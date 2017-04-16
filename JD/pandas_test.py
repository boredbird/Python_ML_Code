import pandas as pd
import numpy as np


df = pd.DataFrame({'total_bill': [16.99, 10.34, 23.68, 23.68, 24.59],
                   'tip': [1.01, 1.66, 3.50, 3.31, 3.61],
                   'sex': ['Female', 'Male', 'Male', 'Male', 'Female']})

# # data type of columns
# print df.dtypes
# # indexes
# print df.index
# # return pandas.Index
# print df.columns
# # each row, return array[array]
print df.values

print df.loc[1:3, ['total_bill', 'tip']]
print df.loc[1:3, 'tip': 'total_bill']
# print df.iloc[1:3, [1, 2]]
# print df.iloc[1:3, 1: 3]