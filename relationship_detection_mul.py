from visual_relationship_dataset import *
import os
import scipy.io as sio
from PIL import Image
import copy
from refine import refine_equiv
from judgeRCC import *
import csv
import re

theta = 0.95
theta_count = 0
np.set_printoptions(precision=2)
np.set_printoptions(threshold=np.inf)

# swith between GPU and CPU
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
weight_dict = dict(zip(selected_predicates, prior_freq))

image_path = sio.loadmat('Visual-Relationship-Detection-master/data/imagePath.mat')
object_detection = sio.loadmat('Visual-Relationship-Detection-master/data/objectDetRCNN.mat')
detection_bboxes = object_detection['detection_bboxes']
detection_labels = object_detection['detection_labels']
detection_confs = object_detection['detection_confs']

embedding_index = {}

with open('ckg_KB_wc_mul_noto.txt', 'r', newline='') as file:
    myreader = csv.reader(file, delimiter=',')
    for rows in myreader:
        if len(rows[1]) > 10:
            reg = re.compile(' +')
            row = reg.sub(' ', (rows[1][1: -1]).strip().replace('\n ', ' ')).split(' ')
            if 'e' in row[0]:
                embedding_index[rows[0]] = [float(s.split('e')[0]) * 10 ** int(s.split('e')[1]) for s in row]
            else:
                embedding_index[rows[0]] = [float(s) for s in row]
for img_id in range(len(detection_bboxes[0])):
    if len(detection_bboxes[0][img_id]) > 0:
        assert np.all(detection_bboxes[0][img_id][:, 0] < detection_bboxes[0][img_id][:, 2])
        assert np.all(detection_bboxes[0][img_id][:, 1] < detection_bboxes[0][img_id][:, 3])

features_detected_bb = []
obj_bboxes_ours = []
sub_bboxes_ours = []
obj_labels_ours = []
sub_labels_ours = []
relationship_embedding_ours = []
sub_obj_label = []
semantic_feat_vect = np.zeros(len(types))
# semantic_feat_vect = np.zeros(100)
rcc_labels = {}
rcc_label_id = {}

with open('rcc_constrain_ori.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        aa = row[2][1:]
        aa = aa[:-1]
        rcc_label_id[row[0] + row[1]] = aa
        aa = row[3][1:]
        aa = aa[:-1]
        rcc_labels[row[0] + row[1]] = row[3]


def np_softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def semantic_Loss(prob, idx, p1, p2):
    one_situation = np.zeros(70)
    for i in idx:
        if (i != ' ') & (i != '\n'):
            one_situation[int(i)] = 1
    loss = np.zeros(70)
    for j in idx:
        if (j != ' ') & (j != '\n'):
            # if prob[int(j)] > 0.1:
            loss[int(j)] = np.log(p1 * p2 * weight_dict[selected_predicates[int(j)]])
            loss[int(j)] = p1 * p2 * np.log(np.abs(np.prod(one_situation - prob)))
    return np_softmax(prob * loss)


def compare_np(a, b):
    bool = True
    for i, it in enumerate(a):
        if a[i] != b[i]:
            bool = False
            break
    return bool


def updateProb(values_of_predicates, sub_obj_pair, label):
    # print('=============================================================')
    # print(values_of_predicates)
    losses = []
    global theta_count, firstPair
    for i, pair in enumerate(sub_obj_pair):
        loss_per_pair = np.zeros(70)
        for midEntity in label:
            if (midEntity != pair[0]) & (midEntity != pair[1]):
                for j, aa in enumerate(sub_obj_pair):
                    if compare_np(aa, [int(pair[0]), int(midEntity)]):
                        firstPair = j
                        break
                for k, bb in enumerate(sub_obj_pair):
                    if compare_np(bb, [int(midEntity), int(pair[1])]):
                        secondPair = k
                        break
                vv = sorted(np.concatenate((values_of_predicates[firstPair], values_of_predicates[secondPair])),reverse=True)
                theta = vv[5]
                max_index1 = np.array(np.where(values_of_predicates[firstPair] > theta))[0]
                theta_count += len(max_index1)
                max_index2 = np.array(np.where(values_of_predicates[secondPair] > theta))[0]

                theta_count += len(max_index2)
                if (len(max_index1) > 0) & (len(max_index2) > 0):
                    preds1 = predicates[max_index1]
                    preds2 = predicates[max_index2]
                    for ii, pred1 in enumerate(preds1):
                        for jj, pred2 in enumerate(preds2):
                            if [pred1 + pred2][0] in rcc_labels.keys():
                                negative_label_id = rcc_label_id[[pred1 + pred2][0]]
                                loss_per_pair += semantic_Loss(values_of_predicates[i], negative_label_id,
                                                               values_of_predicates[firstPair][max_index1[ii]],
                                                               values_of_predicates[secondPair][max_index2[jj]])
        loss_per_pair = np.divide(loss_per_pair, len(label))
        losses.append(loss_per_pair)
    # print(values_of_predicates - np.array(losses))
    # print('=============================================================')
    return values_of_predicates + np.array(losses)


model_list = [
    "models/KB_wc_all_2500.ckpt", "models/KB_wc_mul_2500.ckpt", "models/KB_wc_rcc_2500.ckpt"]

for model in model_list:
    embedding_index = {}
    obj_bboxes_ours_output = []
    sub_bboxes_ours_output = []
    model_label = model.split("/")[-1][:-5]
    print(model.upper())
    with open('ckg_' + model.split("/")[-1].split('_2500')[0] + '.txt', 'r', newline='') as file:
        myreader = csv.reader(file, delimiter=',')
        for rows in myreader:
            if len(rows[1]) > 10:
                reg = re.compile(' +')
                row = reg.sub(' ', (rows[1][1: -1]).strip().replace('\n ', ' ')).split(' ')
                if 'e' in row[0]:
                    embedding_index[rows[0]] = [float(s.split('e')[0]) * 10 ** int(s.split('e')[1]) for s in row]
                else:
                    embedding_index[rows[0]] = [float(s) for s in row]
    for img_id in range(len(detection_bboxes[0])):
        if len(detection_bboxes[0][img_id]) > 0:
            assert np.all(detection_bboxes[0][img_id][:, 0] < detection_bboxes[0][img_id][:, 2])
            assert np.all(detection_bboxes[0][img_id][:, 1] < detection_bboxes[0][img_id][:, 3])
    for pic_idx in range(detection_bboxes.shape[1]):
        detection_bboxes[0, pic_idx] = detection_bboxes[0, pic_idx].astype(np.float)
        features_per_image = np.empty((0, 2 * number_of_features + number_of_extra_features))
        obj_bboxes_ours_per_image = np.array([]).reshape(0, 4)
        sub_bboxes_ours_per_image = np.array([]).reshape(0, 4)
        obj_label_per_image = np.array([])
        sub_label_per_image = np.array([])
        relationship_embedding_per_image = []

        # normalize data
        if len(detection_bboxes[0, pic_idx]) > 0:
            img = Image.open(os.path.join(img_dir, image_path['imagePath'][0, pic_idx][0]).replace('png', 'jpg'))
            width, height = img.size
            normalized_detection_bboxes = copy.deepcopy(detection_bboxes)
            normalized_detection_bboxes[0, pic_idx][:, -4] /= width
            normalized_detection_bboxes[0, pic_idx][:, -3] /= height
            normalized_detection_bboxes[0, pic_idx][:, -2] /= width
            normalized_detection_bboxes[0, pic_idx][:, -1] /= height
        sub_obj_per_image = []
        for bb1_idx in range(len(detection_bboxes[0, pic_idx])):
            for bb2_idx in range(len(detection_bboxes[0, pic_idx])):
                if bb1_idx != bb2_idx:
                    bb1 = normalized_detection_bboxes[0, pic_idx][bb1_idx]
                    bb2 = normalized_detection_bboxes[0, pic_idx][bb2_idx]
                    sub_label_per_image = np.append(sub_label_per_image, detection_labels[0, pic_idx][bb1_idx, 0])
                    obj_label_per_image = np.append(obj_label_per_image, detection_labels[0, pic_idx][bb2_idx, 0])

                    feat_vect_bb1 = np.hstack((semantic_feat_vect, bb1))
                    feat_vect_bb2 = np.hstack((semantic_feat_vect, bb2))
                    feat_vect_bb1[detection_labels[0, pic_idx][bb1_idx]] = detection_confs[0, pic_idx][bb1_idx]
                    feat_vect_bb2[detection_labels[0, pic_idx][bb2_idx]] = detection_confs[0, pic_idx][bb2_idx]
                    feat_vec_pair = np.hstack((feat_vect_bb1, feat_vect_bb2, computing_extended_features(bb1, bb2)))

                    features_per_image = np.vstack((features_per_image, feat_vec_pair[np.newaxis, :]))
                    sub_bboxes_ours_per_image = np.vstack(
                        (sub_bboxes_ours_per_image, detection_bboxes[0, pic_idx][bb1_idx]))
                    obj_bboxes_ours_per_image = np.vstack(
                        (obj_bboxes_ours_per_image, detection_bboxes[0, pic_idx][bb2_idx]))

                    _, rcc_label = judgercc(feat_vec_pair)
                    rcc_embedding = np.zeros(8)
                    rcc_embedding[rcc_label] = 1
                    relationship_embedding_per_image.append(
                        np.hstack([embedding_index[types[detection_labels[0, pic_idx][bb1_idx, 0]]],
                                   embedding_index[types[detection_labels[0, pic_idx][bb2_idx, 0]]], rcc_embedding]))
                    sub_obj_per_image.append(
                        [int(detection_labels[0, pic_idx][bb1_idx]), int(detection_labels[0, pic_idx][bb2_idx])])
        features_detected_bb.append(features_per_image)
        obj_bboxes_ours.append(obj_bboxes_ours_per_image)
        sub_bboxes_ours.append(sub_bboxes_ours_per_image)
        obj_labels_ours.append(obj_label_per_image)
        sub_labels_ours.append(sub_label_per_image)
        relationship_embedding_ours.append(relationship_embedding_per_image)
        sub_obj_label.append(sub_obj_per_image)
    predicted_predicates_values_tensor = tf.concat(
        [isInRelation[predicate].tensor() for predicate in selected_predicates], 1)
    saver = tf.train.Saver()
    sess = tf.Session(config=config)
    saver.restore(sess, model)
    rlp_confs_ours = []
    rlp_labels_ours = []

    for pic_idx in range(detection_bboxes.shape[1]):
        theta_count = 0
        if pic_idx % 100 == 0:
            print("Eval img", pic_idx)
        values_of_predicates = np.array([], dtype=np.float32).reshape(0, 70)
        values_of_predicates = sess.run(predicted_predicates_values_tensor,
                                        {pairs_of_objects.tensor: features_detected_bb[pic_idx]})

        if len(values_of_predicates) > 0:
            values_of_predicates = updateProb(values_of_predicates, sub_obj_label[pic_idx],
                                              detection_labels[0][pic_idx])
        ckg_prob = []
        for n, aa in enumerate(relationship_embedding_ours[pic_idx]):
            bb = sess.run(MLP.pred, {MLP.x: np.expand_dims(aa, 0)})[0]
            ckg_prob.append(bb)
        values_of_predicates = np.reshape(sess.run(results, {F: np.reshape(values_of_predicates,
                                                                           [len(values_of_predicates), 1, 1, 70]),
                                                             P: np.reshape(np.array(ckg_prob),
                                                                           [len(ckg_prob), 1, 1, 70])}),
                                          [len(values_of_predicates), 70])

        values_of_predicates = refine_equiv(values_of_predicates, selected_predicates, "max")
        values_of_predicates = np.multiply(values_of_predicates, prior_freq)
        conf_predicates_per_image = values_of_predicates.flatten('F')
        sub_bboxes_ours_output.append(np.tile(sub_bboxes_ours[pic_idx], (len(selected_predicates), 1)))
        obj_bboxes_ours_output.append(np.tile(obj_bboxes_ours[pic_idx], (len(selected_predicates), 1)))
        label_predicates_per_image = np.hstack(
            (np.tile(sub_labels_ours[pic_idx], len(selected_predicates))[:, np.newaxis],
             np.repeat(np.array(range(1, len(selected_predicates) + 1)), len(features_detected_bb[pic_idx]))[:,
             np.newaxis],
             np.tile(obj_labels_ours[pic_idx], len(selected_predicates))[:, np.newaxis]))

        rlp_confs_ours.append(conf_predicates_per_image[:, np.newaxis])
        rlp_labels_ours.append(label_predicates_per_image)

    sess.close()

    sio.savemat(
        "Visual-Relationship-Detection-master/results/relationship_det_result_" + model_label + ".mat",
        {'sub_bboxes_ours': sub_bboxes_ours_output,
         'obj_bboxes_ours': obj_bboxes_ours_output,
         'rlp_confs_ours': rlp_confs_ours,
         'rlp_labels_ours': rlp_labels_ours})
