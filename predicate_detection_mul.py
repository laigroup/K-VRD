import numpy as np

from visual_relationship_dataset import *
import os
import scipy.io as sio
from PIL import Image
import copy
from refine import refine_equiv
from judgeRCC import *
import csv
import re

np.set_printoptions(precision=2)  # 控制输出结果的精度
np.set_printoptions(threshold=np.inf)  # numpy对数组长度设置了一个阈值，数组长度<=阈值：完整打印；数组长度>阈值：以省略的形式打印；

config = tf.ConfigProto(device_count={'GPU': 1})

img_dir = 'data/sg_test_images'

# Load training data for prio statistics on the dataset
train_data, pairs_of_train_data, types_of_train_data, triples_of_train_data, cartesian_of_train_data, _, cartesian_of_bb_idxs \
    , _, _, idx_types_of_data = get_data("train", True)

idxs_of_positive_examples_of_predicates = {}

for predicate in selected_predicates:
    idxs_of_positive_examples_of_predicates[predicate] = \
        np.where(predicates[triples_of_train_data[:, -1]] == predicate)[0]

prior_stats = np.array([len(idxs_of_positive_examples_of_predicates[pred]) for pred in selected_predicates])
prior_freq = np.true_divide(prior_stats, np.sum(prior_stats))
prior_weights = prior_freq
ckg_weights = np.ones(70) - prior_freq
for n, m in enumerate(prior_freq):
    if m < 0.0001:
        prior_weights[n] = 0
        ckg_weights[n] = 1


image_path = sio.loadmat('Visual-Relationship-Detection-master/data/imagePath.mat')
gt = sio.loadmat('Visual-Relationship-Detection-master/evaluation/gt.mat')
gt_sub_bboxes = gt['gt_sub_bboxes']  # subj的边界框坐标
gt_obj_bboxes = gt['gt_obj_bboxes']  # obj的边界框坐标
gt_tuple_label = gt['gt_tuple_label']  # 三元组（sub_class序号、obj_class序号、谓词序号）1000张图片的，每个图片的三元组放一起

features_detected_bb = []
obj_bboxes_ours = []
sub_bboxes_ours = []
obj_labels_ours = []
sub_labels_ours = []
relationship_embedding_ours = []

semantic_feat_vect = np.zeros(len(types))
# semantic_feat_vect = np.zeros(100)

model_list = [
    "models/KB_wc_all_2500.ckpt", "models/KB_wc_mul_2500.ckpt", "models/KB_wc_rcc_2500.ckpt"]

for model in model_list:
    model_label = model.split("/")[-1][:-5]
    embedding_index = {}
    print(model.upper())
    with open('ckg_'+model.split("/")[-1].split('_2500')[0]+'.txt', 'r', newline='') as file:
        myreader = csv.reader(file, delimiter=',')
        for rows in myreader:
            if len(rows[1]) > 10:
                reg = re.compile(' +')
                row = reg.sub(' ', (rows[1][1: -1]).strip().replace('\n ', ' ')).split(' ')
                if 'e' in row[0]:
                    embedding_index[rows[0]] = [float(s.split('e')[0]) * 10 ** int(s.split('e')[1]) for s in row]
                else:
                    embedding_index[rows[0]] = [float(s) for s in row]
    for img_id in range(len(gt_sub_bboxes[0])):
        if len(gt_sub_bboxes[0][img_id]) > 0:
            assert np.all(gt_sub_bboxes[0][img_id][:, 0] < gt_sub_bboxes[0][img_id][:, 2])
            assert np.all(gt_sub_bboxes[0][img_id][:, 1] < gt_sub_bboxes[0][img_id][:, 3])

            assert np.all(gt_obj_bboxes[0][img_id][:, 0] < gt_obj_bboxes[0][img_id][:, 2])
            assert np.all(gt_obj_bboxes[0][img_id][:, 1] < gt_obj_bboxes[0][img_id][:, 3])



    for pic_idx in range(gt_tuple_label.shape[1]):
        gt_sub_bboxes[0, pic_idx] = gt_sub_bboxes[0, pic_idx].astype(np.float)
        gt_obj_bboxes[0, pic_idx] = gt_obj_bboxes[0, pic_idx].astype(np.float)
        features_per_image = np.empty((0, 2 * number_of_features + number_of_extra_features))
        obj_bboxes_ours_per_image = np.array([]).reshape(0, 4)
        sub_bboxes_ours_per_image = np.array([]).reshape(0, 4)
        obj_label_per_image = np.array([])
        sub_label_per_image = np.array([])
        relationship_embedding_per_image = []

        # normalize data
        if len(gt_sub_bboxes[0, pic_idx]) > 0:
            img = Image.open(os.path.join(img_dir, image_path['imagePath'][0, pic_idx][0]).replace('png', 'jpg'))
            width, height = img.size
            normalized_gt_sub_bboxes = copy.deepcopy(gt_sub_bboxes[0, pic_idx])
            normalized_gt_sub_bboxes[:, -4] /= width
            normalized_gt_sub_bboxes[:, -3] /= height
            normalized_gt_sub_bboxes[:, -2] /= width
            normalized_gt_sub_bboxes[:, -1] /= height

            normalized_gt_obj_bboxes = copy.deepcopy(gt_obj_bboxes[0, pic_idx])
            normalized_gt_obj_bboxes[:, -4] /= width
            normalized_gt_obj_bboxes[:, -3] /= height
            normalized_gt_obj_bboxes[:, -2] /= width
            normalized_gt_obj_bboxes[:, -1] /= height

        for bb_idx in range(len(gt_tuple_label[0, pic_idx])):
            bb1 = normalized_gt_sub_bboxes[bb_idx]
            bb2 = normalized_gt_obj_bboxes[bb_idx]
            sub_label_per_image = np.append(sub_label_per_image, gt_tuple_label[0, pic_idx][bb_idx, 0])
            obj_label_per_image = np.append(obj_label_per_image, gt_tuple_label[0, pic_idx][bb_idx, 2])

            feat_vect_bb1 = np.hstack((semantic_feat_vect, bb1))
            feat_vect_bb2 = np.hstack((semantic_feat_vect, bb2))
            feat_vect_bb1[gt_tuple_label[0, pic_idx][bb_idx, 0]] = 1.0
            feat_vect_bb2[gt_tuple_label[0, pic_idx][bb_idx, 2]] = 1.0

            feat_vec_pair = np.hstack((feat_vect_bb1, feat_vect_bb2, computing_extended_features(bb1, bb2)))
            features_per_image = np.vstack((features_per_image, feat_vec_pair[np.newaxis, :]))
            sub_bboxes_ours_per_image = np.vstack((sub_bboxes_ours_per_image, gt_sub_bboxes[0, pic_idx][bb_idx]))
            obj_bboxes_ours_per_image = np.vstack((obj_bboxes_ours_per_image, gt_obj_bboxes[0, pic_idx][bb_idx]))
            _, rcc_label = judgercc(feat_vec_pair)
            rcc_embedding = np.zeros(8)
            rcc_embedding[rcc_label] = 1
            relationship_embedding_per_image.append(
                np.hstack([embedding_index[types[gt_tuple_label[0, pic_idx][bb_idx, 0]]],
                           embedding_index[types[gt_tuple_label[0, pic_idx][bb_idx, 2]]], rcc_embedding]))
        features_detected_bb.append(features_per_image)
        obj_bboxes_ours.append(obj_bboxes_ours_per_image)
        sub_bboxes_ours.append(sub_bboxes_ours_per_image)
        obj_labels_ours.append(obj_label_per_image)
        sub_labels_ours.append(sub_label_per_image)
        relationship_embedding_ours.append(relationship_embedding_per_image)

    obj_bboxes_ours_output = []
    sub_bboxes_ours_output = []
    predicted_predicates_values_tensor = tf.concat(
        [isInRelation[predicate].tensor() for predicate in selected_predicates], 1)
    saver = tf.train.Saver()
    sess = tf.Session(config=config)
    saver.restore(sess, model)
    rlp_confs_ours = []
    rlp_labels_ours = []
    error_an = np.array([]).reshape(0, 8)
    for pic_idx in range(gt_tuple_label.shape[1]):
        if pic_idx % 100 == 0:
            print("Eval img", pic_idx)
        values_of_predicates = sess.run(predicted_predicates_values_tensor,
                                        {pairs_of_objects.tensor: features_detected_bb[pic_idx]})

        ckg_prob = []
        for n, aa in enumerate(relationship_embedding_ours[pic_idx]):
            bb = sess.run(MLP.pred, {MLP.x: np.expand_dims(aa, 0)})[0]
            ckg_prob.append(bb)
        ckg_prob = np.array(ckg_prob)
        values_of_predicates =values_of_predicates + np.multiply(np.reshape(
            sess.run(results, {F: np.reshape(np.multiply(values_of_predicates, prior_weights),
                                             [len(values_of_predicates), 1, 1, 70]),
                               P: np.reshape(ckg_prob, [len(ckg_prob), 1, 1, 70])}),
            [len(values_of_predicates), 70]), prior_freq)
        values_of_predicates = refine_equiv(values_of_predicates, selected_predicates, "max")
        values_of_predicates = np.multiply(values_of_predicates, prior_freq)
        conf_predicates_per_image = values_of_predicates.flatten('F')
        sub_bboxes_ours_output.append(np.tile(sub_bboxes_ours[pic_idx], (len(selected_predicates), 1)))
        obj_bboxes_ours_output.append(np.tile(obj_bboxes_ours[pic_idx], (len(selected_predicates), 1)))
        # Matlab indices start from 1
        label_predicates_per_image = np.hstack(
            (np.tile(sub_labels_ours[pic_idx], len(selected_predicates))[:, np.newaxis],
             np.repeat(np.array(range(1, len(selected_predicates) + 1)), len(features_detected_bb[pic_idx]))[:,
             np.newaxis],
             np.tile(obj_labels_ours[pic_idx], len(selected_predicates))[:, np.newaxis]))

        rlp_confs_ours.append(conf_predicates_per_image[:, np.newaxis])
        rlp_labels_ours.append(label_predicates_per_image)
    sess.close()
    sio.savemat("Visual-Relationship-Detection-master/results/predicate_det_result_" + model_label + ".mat",
                {'sub_bboxes_ours': sub_bboxes_ours_output,
                 'obj_bboxes_ours': obj_bboxes_ours_output,
                 'rlp_confs_ours': rlp_confs_ours,
                 'rlp_labels_ours': rlp_labels_ours})
