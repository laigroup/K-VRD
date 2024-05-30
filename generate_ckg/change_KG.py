import numpy as np
import pandas as pd
import csv
import tensorflow as tf

def changeCKG(FILE, sys_dict):
    pd.set_option('display.max_rows', None)
    # 最后的的参数可以限制输出行的数量

    # 设置列不限制数量
    pd.set_option('display.max_columns', None)
    # 最后的的参数可以限制输出列的数量

    # 设置value的显示长度为100，默认为50
    pd.set_option('max_colwidth', 100000)

    # 删除不只含有中文节点的关系
    data = pd.read_csv(FILE + '.csv', encoding='ISO-8859-1')

    with open(r'chang' + FILE + 'toTripet.csv', mode='w', newline='', encoding='utf8') as cf:
        wf = csv.writer(cf)
        for index, d in data.iterrows():
            wf.writerow([d['start'].split('/')[3], d['relation'].split('/')[2], d['end'].split('/')[3], float(d['weights'])/10])
        for ii in sys_dict.keys():
            for ss in sys_dict[ii].sys:
                print(ss)
                wf.writerow([ii, 'SimilarTo', ss, sys_dict[ii].sys[ss]])
    cf.close()
