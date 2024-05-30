import csv
import pickle as cp
import numpy as np
import os
import tensorflow as tf
import csv

ontology_dir = "data/ontology"
object_cate_num = 101
predicate_cate_num = 71
data_dir = "data/train"


def getCKG(idxs_of_positive_examples_of_predicates):
    print("build CKG")
    data_dir = "data/train"
    triples_s_o_p = np.genfromtxt(os.path.join(data_dir, "predicates.csv"), delimiter=",", dtype="i", max_rows=10000000)
    idx_types_of_data = np.genfromtxt(os.path.join(data_dir, "types.csv"), dtype="i", max_rows=10000000)
    data_dir = "data/ontology"
    subs = np.genfromtxt(os.path.join(data_dir, "classes.csv"), delimiter=",", dtype="str", max_rows=10000000)
    objs = np.genfromtxt(os.path.join(data_dir, "classes.csv"), delimiter=",", dtype="str", max_rows=10000000)
    preds = np.genfromtxt(os.path.join(data_dir, "predicates.csv"), delimiter=",", dtype="str", max_rows=10000000)
    types_of_data = subs[idx_types_of_data]

    prob = {}
    count = {}
    ckg_pro = {}
    for sub in subs:  # 0~100
        prob[sub] = {}
        count[sub] = {}
        ckg_pro[sub] = {}
        for obj in objs:
            prob[sub][obj] = {}
            count[sub][obj] = {}
            ckg_pro[sub][obj] = {}
            for pred in preds:  # 0~70
                prob[sub][obj][pred] = 0
                count[sub][obj][pred] = 0

    num = 0
    continue_bool1 = True
    continue_bool2 = True
    print(preds)
    for pred in preds:
        num = len(idxs_of_positive_examples_of_predicates[pred])

        for id in idxs_of_positive_examples_of_predicates[pred]:
            aa = types_of_data[triples_s_o_p[id][0]]
            bb = types_of_data[triples_s_o_p[id][1]]
            cc = preds[triples_s_o_p[id][2]]
            print(aa, cc, bb)
            count[aa][bb][cc] = count[aa][bb][cc] + 1
            prob[aa][bb][cc] = (count[aa][bb][cc]) * 1.0 / (num * 1.0)

    for sub in subs:  # 0~100
        for obj in objs:
            ckg_pro[sub][obj] = list(prob[sub][obj].values())

    print("compute done")
    print("write pkl...")
    with  open('D:/dn/code/Visual-Relationship-Detection-LTN-master/CKG_vrd_pro.pkl', 'wb') as fid:
        cp.dump(ckg_pro, fid, cp.HIGHEST_PROTOCOL)

    print("CKG ready")


#
predicates = np.genfromtxt(os.path.join(ontology_dir, "predicates.csv"), dtype="str", delimiter=",", max_rows=100000)
triples_of_train_data = np.genfromtxt(os.path.join(data_dir, "predicates.csv"), delimiter=",", dtype="i",
                                      max_rows=100000)
selected_predicates = predicates

idxs_of_positive_examples_of_predicates = {}

idx_of_cleaned_data = np.where(
    np.in1d(predicates[triples_of_train_data[:, -1]], selected_predicates))  # 把不在onology中的谓词排除

triples_s_o_p = triples_of_train_data[idx_of_cleaned_data]

for predicate in selected_predicates:
    idxs_of_positive_examples_of_predicates[predicate] = \
        np.where(predicates[triples_of_train_data[:, -1]] == predicate)[0]

getCKG(idxs_of_positive_examples_of_predicates)
