from visual_relationship_dataset import *
import tensorflow as tf
import logictensornetworks as ltn
import numpy as np
from multi_hop import *
from mlp import *
from judgeRCC import *

ltn.default_optimizer = "rmsprop"

# swith between GPU and CPU
config = tf.ConfigProto(device_count={'GPU': 1})

number_of_positive_examples_types = 100
number_of_negative_examples_types = 100
number_of_positive_examples_predicates = 100
number_of_negative_examples_predicates = 100

# Load training data
train_data, pairs_of_train_data, types_of_train_data, triples_of_train_data, cartesian_of_train_data, _, cartesian_of_bb_idxs \
    , _, _, idx_types_of_data = get_data("train", True)
set_triples_of_train_data = set([(bb_pairs_idx[0], bb_pairs_idx[1]) for bb_pairs_idx in triples_of_train_data[:, :2]])
idxs_of_negative_examples = [idx for idx, pair in enumerate(cartesian_of_bb_idxs) if
                             tuple(pair) not in set_triples_of_train_data]

# Computing positive and negative examples for predicates and types
idxs_of_positive_examples_of_predicates = {}
idxs_of_negative_examples_of_predicates = {}
idxs_of_positive_examples_of_types = {}

for type in selected_types:
    idxs_of_positive_examples_of_types[type] = np.where(types_of_train_data == type)[0]

for predicate in selected_predicates:
    idxs_of_positive_examples_of_predicates[predicate] = \
        np.where(predicates[triples_of_train_data[:, -1]] == predicate)[0]
    idxs_of_negative_examples_of_predicates[predicate] = idxs_of_negative_examples

print("finished to upload and analyze data")
print("Start model definition")

for predicate in selected_predicates:
    idxs_of_positive_examples_of_predicates[predicate] = \
    np.where(predicates[triples_of_train_data[:, -1]] == predicate)[0]

prior_stats = np.array([len(idxs_of_positive_examples_of_predicates[pred]) for pred in selected_predicates])
prior_freq = np.true_divide(prior_stats, np.sum(prior_stats))
weight_dict = dict(zip(selected_predicates, prior_freq))

predicted_predicates_values_tensor = tf.concat(
    [isInRelation[predicate].tensor() for predicate in selected_predicates], 1)



train_learning_rate = 0.0001
global_step = tf.Variable(0, trainable=False)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=L, logits=tf.reshape(results, [100, 70])),
                      name='loss')

trainable_scope = [
    'P/f_update/weights',
    'P/f_update/biases',
    'P/p_update/weights',
    'P/p_update/biases'
]

scopes = trainable_scope

variables_to_train = []
for scope in scopes:
    variables = tf.trainable_variables(scope)
    variables_to_train.extend(variables)

var_list = variables_to_train

learning_rate = tf.train.exponential_decay(train_learning_rate, global_step, 1000, 0.7, staircase='True',
                                           name='learning_rate')  # 0.7

train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# domain definition
clause_for_positive_examples_of_predicates = [
    ltn.Clause([ltn.Literal(True, isInRelation[p], object_pairs_in_relation[p])],
               label="examples_of_object_pairs_in_" + p.replace(" ", "_") + "_relation") for p in
    selected_predicates]

clause_for_negative_examples_of_predicates = [
    ltn.Clause([ltn.Literal(False, isInRelation[p], object_pairs_not_in_relation[p])],
               label="examples_of_object_pairs_not_in_" + p.replace(" ", "_") + "_relation") for p in
    selected_predicates]

# axioms from the Visual Relationship Ontology
isa_subrelation_of, has_subrelations, inv_relations_of, not_relations_of, reflexivity_relations, symmetry, domain_relation, range_relation = get_vrd_ontology()

so_domain = {}
os_domain = {}

for type in selected_types:
    so_domain[type] = ltn.Domain(number_of_features * 2 + number_of_extra_features, label="object_pairs_for_axioms")
    os_domain[type] = ltn.Domain(number_of_features * 2 + number_of_extra_features,
                                 label="inverse_object_pairs_for_axioms")

clauses_for_not_domain = [ltn.Clause([ltn.Literal(False, isInRelation[pred], so_domain[subj[4:]]),
                                      ltn.Literal(False, isOfType[subj[4:]], objects_of_type[subj[4:]])],
                                     label="not_domain_of_" + pred.replace(" ", "_"), weight=weight_dict[p])
                          for pred in domain_relation.keys() for subj in domain_relation[pred]
                          if len(domain_relation[pred]) > 0 and subj.split(" ")[0] == "not"]

clauses_for_not_range = [ltn.Clause([ltn.Literal(False, isInRelation[pred], os_domain[obj[4:]]),
                                     ltn.Literal(False, isOfType[obj[4:]], objects_of_type[obj[4:]])],
                                    label="not_range_of_" + pred.replace(" ", "_"), weight=weight_dict[p])
                         for pred in range_relation.keys() for obj in range_relation[pred]
                         if len(range_relation[pred]) > 0 and obj.split(" ")[0] == "not"]


def train(number_of_training_iterations,
          frequency_of_feed_dict_generation,
          with_constraints,
          start_from_iter=1,
          saturation_limit=0.90):
    global idxs_of_positive_examples_of_predicates, idxs_of_negative_examples_of_predicates

    # defining the clauses of the background knowledge
    clauses = clause_for_positive_examples_of_predicates + clause_for_negative_examples_of_predicates

    if with_constraints:
        clauses = clauses + \
                  clauses_for_not_domain + \
                  clauses_for_not_range

    # defining the label of the background knowledge

    if with_constraints:
        kb_label = "KB_wc_mul"
    else:
        kb_label = "KB_nc_mul"

    models_path = "models/"
    KB = ltn.KnowledgeBase(kb_label, clauses, models_path)
    # start training
    init = tf.initialize_all_variables()
    sess = tf.Session(config=config)

    if start_from_iter == 1:
        sess.run(init)
    if start_from_iter > 1:
        KB.restore(sess)

    feed_dict, pos_eg_id = get_feed_dict(idxs_of_positive_examples_of_predicates,
                                         idxs_of_negative_examples_of_predicates,
                                         idxs_of_positive_examples_of_types, with_constraints=with_constraints)
    train_kb = True

    for i in range(start_from_iter, number_of_training_iterations + 1):
        if i % frequency_of_feed_dict_generation == 0:
            if i / frequency_of_feed_dict_generation > 1:
                pair_idx = {}
                for pre_id in pos_eg_id.keys():
                    feature_per_pred = np.empty((0, 2 * number_of_features + number_of_extra_features))
                    ltn_prob = np.empty((0, 70), dtype=np.float32)
                    label = np.zeros(70)
                    label[pre_id] = 1.0
                    labels = np.empty((0, 70), dtype=np.float32)
                    pair_idx[pre_id] = []
                    rcc = []
                    for s_o_id in pos_eg_id[pre_id]:
                        rcc_vector = np.zeros(8)
                        feature_per_pred = np.vstack((feature_per_pred, pairs_of_train_data[s_o_id]))
                        pair_idx[pre_id].append([idx_types_of_data[triples_of_train_data[s_o_id][0]],
                                                 idx_types_of_data[triples_of_train_data[s_o_id][1]]])
                        _, loc = judgercc(pairs_of_train_data[s_o_id])
                        rcc_vector[loc] = 1
                        rcc.append(rcc_vector)
                        labels = np.vstack([labels, label])
                    values_of_predicates = sess.run(predicted_predicates_values_tensor,
                                                    {pairs_of_objects.tensor: feature_per_pred})
                    mlp_train_data, ckg_weight = aggregate_subgraph(pair_idx[pre_id])
                    ckg_prob = MLP.train(np.concatenate((mlp_train_data, rcc), axis=1), labels, sess)
                    for n, w in enumerate(ckg_weight):
                        ckg_prob[n] = np.add(ckg_prob[n], sess.run(prob, {CP: w}))
                    for j in values_of_predicates:
                        ltn_prob = np.vstack([ltn_prob, j])
                    sess.run(train_step, {F: np.reshape(ltn_prob, [100, 1, 1, 70]),
                                          P: np.reshape(ckg_prob, [100, 1, 1, 70]), L: labels})
            if train_kb:
                print(i)
            else:
                train_kb = True
            if train_kb and (i == number_of_training_iterations):
                KB.save(sess, version="_" + str(i))

            feed_dict, pos_eg_id = get_feed_dict(idxs_of_positive_examples_of_predicates,
                                                 idxs_of_negative_examples_of_predicates,
                                                 idxs_of_positive_examples_of_types,
                                                 with_constraints=with_constraints)
            print("---- TRAIN", kb_label, "----")
        if train_kb:
            sat_level = sess.run(KB.tensor, feed_dict)

            if np.isnan(sat_level):
                train_kb = False
            if sat_level >= saturation_limit:
                train_kb = False
            else:
                KB.train(sess, feed_dict)
        print(str(i) + ' --> ' + str(sat_level))
    write2file(kb_label)
    print("end of training")
    sess.close()


def get_feed_dict(idxs_of_pos_ex_of_predicates, idxs_of_neg_ex_of_predicates, idxs_of_pos_ex_of_types,
                  with_constraints):
    print("selecting new training data")
    feed_dict = {}
    prob_idx = {}

    # positive and negative examples for predicates
    for pred_id, p in enumerate(predicates):
        aa = np.random.choice(idxs_of_pos_ex_of_predicates[p], number_of_positive_examples_predicates)
        feed_dict[object_pairs_in_relation[p].tensor] = pairs_of_train_data[aa]
        prob_idx[pred_id] = aa
        feed_dict[object_pairs_not_in_relation[p].tensor] = \
            cartesian_of_train_data[
                np.random.choice(idxs_of_neg_ex_of_predicates[p], number_of_negative_examples_predicates)]

    # feed data for axioms
    if with_constraints:
        for t in selected_types:
            idxs_bb_type = np.random.choice(idxs_of_pos_ex_of_types[t], number_of_positive_examples_types)
            feed_dict[objects_of_type[t].tensor] = train_data[idxs_bb_type][:, 1:]

            idxs_bb_pairs_subj = []
            idxs_bb_pairs_obj = []

            for idx in idxs_bb_type:
                idxs_bb_pairs_subj.append(np.random.choice(np.where(cartesian_of_bb_idxs[:, 0] == idx)[0], 1)[0])
                idxs_bb_pairs_obj.append(np.random.choice(np.where(cartesian_of_bb_idxs[:, 1] == idx)[0], 1)[0])

            feed_dict[so_domain[t].tensor] = cartesian_of_train_data[idxs_bb_pairs_subj]
            feed_dict[os_domain[t].tensor] = cartesian_of_train_data[idxs_bb_pairs_obj]
    return feed_dict, prob_idx


if __name__ == "__main__":
    train(number_of_training_iterations=2500,
          frequency_of_feed_dict_generation=50,
          with_constraints=True,
          start_from_iter=1,
          saturation_limit=.96)
