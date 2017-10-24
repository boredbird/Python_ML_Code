# -*- coding:utf-8 -*-
import pandas as pd
raw_data_path = r'E:\output\rawdata\ccf_first_round_shop_info.csv'
dataset = pd.read_csv(raw_data_path)

dataset_sub = dataset[dataset['mall_id'] == 'm_4543']


import matplotlib
import matplotlib.pyplot as plt
plt.scatter(dataset_sub['longitude'],dataset_sub['latitude'] ,c=dataset_sub['category_id'])

plt.xlim(dataset_sub['longitude'].min() , dataset_sub['longitude'].max() )
plt.ylim(dataset_sub['latitude'].min() , dataset_sub['latitude'].max() )

for x, y ,z in zip(dataset_sub['longitude'], dataset_sub['latitude'],dataset_sub['mall_id']):
    plt.annotate(
        z,
        xy=(x, y),
        xytext=(0, -5),
        textcoords='offset points',
        ha='center',
        va='top',
        fontsize=8)

plt.show()
