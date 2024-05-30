import os
from scipy.special import softmax
from scipy.sparse import csc_matrix
from tqdm import tqdm
import pickle
import scipy.sparse as ssp
import random
from scipy.sparse.csgraph import dijkstra
import numpy as np
import tensorflow as tf
import csv

ontology_dir = "data/ontology"

types = np.genfromtxt(os.path.join(ontology_dir, "classes.csv"), dtype="str", delimiter=",")
predicates = np.genfromtxt(os.path.join(ontology_dir, "predicates.csv"), dtype="str", delimiter=",")

file = 'changCKG_andtoTripet.csv'
saved_relation2id = None
max_label_value = None


def process_files(file, saved_relation2id=None):
    entity2id = {}
    relation2id = {} if saved_relation2id is None else saved_relation2id
    w_ckg = {}
    ent = 0
    rel = 0

    for id, entity in enumerate(list(types) + list(predicates)):
        entity2id[entity] = id
        ent = id + 1
    data = []
    with open(file) as f:
        file_data = [line.split() for line in f.read().split('\n')[:-1]]
        file_data = [line[0].split(',') for line in file_data]
    for triplet in file_data:
        if triplet[0] not in entity2id:
            entity2id[triplet[0]] = ent
            ent += 1
        if triplet[2] not in entity2id:
            entity2id[triplet[2]] = ent
            ent += 1
        if not saved_relation2id and triplet[1] not in relation2id:
            relation2id[triplet[1]] = rel
            rel += 1
        # Save the triplets corresponding to only the known relations
        if triplet[1] in relation2id:
            data.append([entity2id[triplet[0]], entity2id[triplet[2]], relation2id[triplet[1]], float(triplet[3])])
            if [entity2id[triplet[0]], entity2id[triplet[2]]] not in list(w_ckg.keys()):
                w_ckg[entity2id[triplet[0]], entity2id[triplet[2]]] = float(triplet[3])/10
            else:
                w_ckg[entity2id[triplet[0]], entity2id[triplet[2]]] = max(w_ckg[entity2id[triplet[0]],
                entity2id[triplet[2]]], float(triplet[3])/10)
            if [entity2id[triplet[2]], entity2id[triplet[0]]] not in list(w_ckg.keys()):
                w_ckg[entity2id[triplet[2]], entity2id[triplet[0]]] = float(triplet[3])/10
            else:
                w_ckg[entity2id[triplet[2]], entity2id[triplet[0]]] = max(w_ckg[entity2id[triplet[2]],
                entity2id[triplet[0]]], float(triplet[3])/10)
    triplets = np.array(data)
    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # Construct the list of adjacency matrix each corresponding to eeach relation. Note that this is constructed only
    # from the train data.
    adj_list = []

    for i in range(len(relation2id)):
        idx = np.argwhere(triplets[:, 2] == i)
        adj_list.append(csc_matrix((triplets[:, 3][idx].squeeze(1),
                                    (triplets[:, 0][idx].squeeze(1), triplets[:, 1][idx].squeeze(1))),
                                   shape=(len(entity2id), len(entity2id))))

    nei_node = {}
    for i in entity2id.keys():
        nei_node[int(entity2id[i])] = get_neighbor_nodes({int(entity2id[i])}, adj_list, 2)
    return adj_list, triplets, entity2id, relation2id, id2entity, id2relation, nei_node, w_ckg


def sample_neg(adj_list, edges, num_neg_samples_per_link=1, max_size=1000000, constrained_neg_prob=0):
    pos_edges = edges
    neg_edges = []

    # if max_size is set, randomly sample train links
    if max_size < len(pos_edges):
        perm = np.random.permutation(len(pos_edges))[:max_size]
        pos_edges = pos_edges[perm]

    # sample negative links for train/test
    n, r = adj_list[0].shape[0], len(adj_list)

    # distribution of edges across relations
    theta = 0.001
    edge_count = get_edge_count(adj_list)
    """数组shape = （1，70），稀疏矩阵几条边"""
    rel_dist = np.zeros(edge_count.shape)
    idx = np.nonzero(edge_count)
    """返回非负值的索引"""
    rel_dist[idx] = softmax(theta * edge_count[idx])

    # possible head and tails for each relation
    valid_heads = [adj.tocoo().row.tolist() for adj in adj_list]
    valid_tails = [adj.tocoo().col.tolist() for adj in adj_list]

    pbar = tqdm(total=len(pos_edges))
    while len(neg_edges) < num_neg_samples_per_link * len(pos_edges):
        neg_head, neg_tail, rel = pos_edges[pbar.n % len(pos_edges)][0], pos_edges[pbar.n % len(pos_edges)][1], \
            pos_edges[pbar.n % len(pos_edges)][2]
        if np.random.uniform() < constrained_neg_prob:
            if np.random.uniform() < 0.5:
                neg_head = np.random.choice(valid_heads[rel])
            else:
                neg_tail = np.random.choice(valid_tails[rel])
        else:
            if np.random.uniform() < 0.5:
                neg_head = np.random.choice(n)
            else:
                neg_tail = np.random.choice(n)
        neg_head = int(neg_head)
        neg_tail = int(neg_tail)
        rel = int(rel)
        if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
            neg_edges.append([neg_head, neg_tail, rel])
            pbar.update(1)
    pbar.close()

    neg_edges = np.array(neg_edges)
    return pos_edges, neg_edges


def get_edge_count(adj_list):
    count = []
    for adj in adj_list:
        count.append(len(adj.tocoo().row.tolist()))
    return np.array(count)


def links2subgraphs(A, graphs, nei_node, entity2id, max_label_value=None):
    avg_size_sub2obj, datum = get_average_subgraph_size(100, graphs['sub2obj'], A, nei_node)
    avg_size_sub2obj = avg_size_sub2obj * 1.5
    links_length = 0
    links_length += (len(graphs['pos']) + len(graphs['neg'])) * 2

    return datum


def get_average_subgraph_size(sample_size, links, A, nei_node):
    """                      (100, graphs['pos'], A, params) * 1.5"""
    total_size = 0
    pbar = tqdm(total=len(links))
    subgraphs = []
    datum = {}

    for (n1, n2, r_label) in links:
        nodes, n_labels, subgraph_size, enc_ratio, num_pruned_nodes, subgraph, mid_node = \
            subgraph_extraction_labeling((n1, n2), A, 2, nei_node, True, max_node_label_value=None)
        datum[(n1, n2)] = {'nodes': nodes, 'n_labels': n_labels, 'mid_nodes': mid_node}
        total_size += len(serialize(datum))
        subgraphs.append(subgraph)
        pbar.update(1)
    pbar.close()
    return total_size / sample_size, datum


def subgraph_extraction_labeling(ind, A_list, h, nei_node, enclosing_sub_graph=False, max_nodes_per_hop=None,
                                 max_node_label_value=None):
    # extract the h-hop enclosing subgraphs around link 'ind'

    pruned_subgraph_nodes = None
    pruned_labels = None
    subgraph_size = None
    enc_ratio = None
    num_pruned_nodes = None
    root1_nei = nei_node[(int(ind[0]))]
    root2_nei = nei_node[(int(ind[1]))]

    subgraph_nei_nodes_int = root1_nei.intersection(root2_nei)
    subgraph_nei_nodes_un = root1_nei.union(root2_nei)
    mid_nodes = None
    # Extract subgraph | Roots being in the front is essential for labelling and the model to work properly.
    if enclosing_sub_graph:
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_int)
        """子图的头尾是0，1"""
    else:
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_un)

    subgraph = [adj[subgraph_nodes, :][:, subgraph_nodes]
                for adj in A_list if np.sum(adj[subgraph_nodes, :][:, subgraph_nodes]) != 0]

    if len(subgraph) > 0:
        subgraph = incidence_matrix(subgraph)
        """得到的是[idx1_subgraph_nodes,idx2_subgraph_nodes] relation数"""
        labels, enclosing_subgraph_nodes, mid_nodes = node_label(subgraph_nodes, subgraph, max_distance=3)
        pruned_subgraph_nodes = np.array(subgraph_nodes)[enclosing_subgraph_nodes].tolist()
        pruned_labels = labels[enclosing_subgraph_nodes]
        # pruned_subgraph_nodes = subgraph_nodes
        if max_node_label_value is not None:
            pruned_labels = np.array([np.minimum(label, max_node_label_value).tolist() for label in pruned_labels])

        subgraph_size = len(pruned_subgraph_nodes)
        enc_ratio = len(subgraph_nei_nodes_int) / (len(subgraph_nei_nodes_un) + 1e-3)
        num_pruned_nodes = len(subgraph_nodes) - len(pruned_subgraph_nodes)

    return pruned_subgraph_nodes, pruned_labels, subgraph_size, enc_ratio, num_pruned_nodes, subgraph, mid_nodes


def shortest_path(subgraph, subgraph_nodes, max_distance):
    roots = [0, 1]
    sgs_single_root = [remove_nodes(subgraph, [root]) for root in roots]
    dists = []
    nodes = []
    for r, sg in enumerate(sgs_single_root):
        dist, node = dijkstra(sg, indices=[0], directed=False, unweighted=True, limit=1e6, return_predecessors=True)
        dists.append(np.clip(dist[0][1:], 0, 1e7))
        nodes.append(node[0][1:])
    dists_to_roots = np.array(list(zip(dists[0], dists[1])), dtype=int)
    node_to_roots = np.array(list(zip(nodes[0], nodes[1])), dtype=int)
    for bb, pair in enumerate(node_to_roots):
        if min(pair) > 0:
            node_to_roots[bb] = [subgraph_nodes[int(pair[0]) + 1], subgraph_nodes[int(pair[1]) + 1]]
        elif min(pair) == 0:
            if (pair[0] == 0) & (pair[1] != 0):
                node_to_roots[bb] = [subgraph_nodes[0], subgraph_nodes[int(pair[1]) + 1]]
            if (pair[1] == 0) & (pair[0] != 0):
                node_to_roots[bb] = [subgraph_nodes[int(pair[0]) + 1], subgraph_nodes[1]]
            if (pair[1] == 0) & (pair[0] == 0):
                node_to_roots[bb] = [subgraph_nodes[0], subgraph_nodes[1]]
    target_node_labels = np.array([[0, 1], [1, 0]])
    target_mid_node = np.array([[-1, -1], [-1, -1]])
    labels = np.concatenate((target_node_labels, dists_to_roots)) if dists_to_roots.size else target_node_labels
    nodes_to_roots = np.concatenate((target_mid_node, node_to_roots)) if node_to_roots.size else target_mid_node

    idx_node = np.where(np.max(labels, axis=1) <= max_distance)
    mid_nodes = []
    for id_node in idx_node:
        mid_nodes.append(nodes_to_roots[id_node])
    return labels, mid_nodes


def node_label(subgraph_nodes, subgraph, max_distance=1):
    # implementation of the node labeling scheme described in the paper
    labels, mid_nodes = shortest_path(subgraph, subgraph_nodes, max_distance)
    enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]
    pruned_subgraph_nodes = np.array(subgraph_nodes)[enclosing_subgraph_nodes].tolist()
    for cc, m in enumerate(mid_nodes[0]):
        if (m[0] not in pruned_subgraph_nodes) | (m[1] not in pruned_subgraph_nodes):
            del list(mid_nodes[0])[cc]
            del list(labels)[cc]
            del list(enclosing_subgraph_nodes)[cc]

    return np.array(labels), np.array(enclosing_subgraph_nodes), np.array(mid_nodes[0])


def serialize(data):
    data_tuple = tuple(data.values())
    return pickle.dumps(data_tuple)


def incidence_matrix(adj_list):
    dim = adj_list[0].shape
    maxadj = adj_list[0].todense()

    for adj in adj_list:
        maxadj = np.maximum(maxadj, adj.todense())
    return ssp.csc_matrix(maxadj, shape=dim)


def remove_nodes(A_incidence, nodes):
    idxs_wo_nodes = list(set(range(A_incidence.shape[1])) - set(nodes))
    return A_incidence[idxs_wo_nodes, :][:, idxs_wo_nodes]


def get_neighbor_nodes(roots, adj, h=1, max_nodes_per_hop=None):
    A_incidence = incidence_matrix(adj)
    A_incidence += A_incidence.T
    bfs_generator = _bfs_relational(A_incidence, roots, max_nodes_per_hop)
    lvls = list()
    for _ in range(h):
        try:
            lvls.append(next(bfs_generator))
        except StopIteration:
            pass
    return set().union(*lvls)


def _bfs_relational(adj, roots, max_nodes_per_hop=None):
    visited = set()
    current_lvl = set(roots)

    while current_lvl:

        for v in current_lvl:
            visited.add(v)

        next_lvl = _get_neighbors(adj, current_lvl)
        next_lvl -= visited  # set difference

        if max_nodes_per_hop and max_nodes_per_hop < len(next_lvl):
            next_lvl = set(random.sample(next_lvl, max_nodes_per_hop))

        yield next_lvl

        current_lvl = set.union(next_lvl)


def _get_neighbors(adj, nodes):
    sp_nodes = _sp_row_vec_from_idx_list(list(nodes), adj.shape[1])
    sp_neighbors = sp_nodes.dot(adj)
    """乘法"""
    neighbors = set(ssp.find(sp_neighbors)[1])  # convert to set of indices
    return neighbors


def _sp_row_vec_from_idx_list(idx_list, dim):
    """_sp_row_vec_from_idx_list(set([ind[0]]), adj.shape[1])"""
    """Create sparse vector of dimensionality dim from a list of indices."""
    shape = (1, dim)
    data = np.ones(len(idx_list))
    row_ind = np.zeros(len(idx_list))
    col_ind = list(idx_list)
    return ssp.csr_matrix((data, (row_ind, col_ind)), shape=shape)


def update_embedding(w_ckg, tail_id, head_id, w_rt):
    head_label = (entities[int(head_id)].lower()).replace(' ', '_')
    tail_label = (entities[int(tail_id)].lower()).replace(' ', '_')
    w_edge = w_ckg[int(head_id), int(tail_id)]
    head_embedding = embedding_index[head_label]
    tail_embedding = embedding_index[tail_label]
    update = np.concatenate([head_embedding, tail_embedding])
    update = np.multiply(w_edge * w_rt, update)
    return update


def aggregate_subgraph(idx):
    global entities
    entities = list(entity.keys())
    train_rel_embedding = []
    p_ckg = []
    for id in idx:
        node_f = [id[0], id[1]]
        node_b = [id[0], id[1]]
        node_update_f = np.zeros(300)
        node_update_b = np.zeros(300)
        node_update_f_fin = np.zeros(300)
        node_update_b_fin = np.zeros(300)
        edge_embedding = np.zeros(600)
        data = datum[(id[0], id[1])]
        prob = np.zeros(70)
        count_node = 0
        count_edge = 0
        head_embedding = embedding_index[(types[int(id[0])].lower()).replace(' ', '_')]
        tail_embedding = embedding_index[(types[int(id[1])].lower()).replace(' ', '_')]
        relationship_embedding = np.concatenate((head_embedding, tail_embedding))
        if data['nodes'] is not None:
            for aa, n in enumerate(data['nodes']):
                if n not in node_f:
                    node_f.append(n)
                    if data['n_labels'][aa][0] == 1:
                        count_edge = count_edge + 1
                        update = update_embedding(w_ckg, id[1], n, 0.8)
                        edge_embedding += update
                if n not in node_b:
                    node_b.append(n)
                    if data['n_labels'][aa][1] == 1:
                        count_edge = count_edge + 1
                        update = update_embedding(w_ckg, n, id[0], 0.8)
                        edge_embedding += update
                if data['n_labels'][aa][0] == 2:
                    mid = data['mid_nodes'][aa][0]
                    node_update_f = embedding_index[(entities[int(n)].lower()).replace(' ', '_')] * \
                                    (7 - np.sum(data['n_labels'][aa])) / 6 * w_ckg[id[1], mid] * w_ckg[mid, n]
                    node_f.append(n)
                    if mid not in node_f:
                        node_f.append(mid)
                        count_edge = count_edge + 2
                        update = update_embedding(w_ckg, id[1], mid, 0.8)
                        edge_embedding += update
                        update = update_embedding(w_ckg, mid, n, 0.4)
                        edge_embedding += update
                    else:
                        count_edge = count_edge + 1
                        update = update_embedding(w_ckg, mid, n, 0.4)
                        edge_embedding += update
                if data['n_labels'][aa][1] == 2:
                    mid = data['mid_nodes'][aa][1]
                    node_update_b = embedding_index[(entities[int(n)].lower()).replace(' ', '_')] * \
                                    ((7 - np.sum(data['n_labels'][aa])) / 6) * w_ckg[mid, id[0]] * w_ckg[n, mid]
                    node_b.append(n)
                    if mid not in node_b:
                        count_edge = count_edge + 2
                        node_b.append(mid)
                        update = update_embedding(w_ckg, mid, id[0], 0.8)
                        edge_embedding += update
                        update = update_embedding(w_ckg, n, mid, 0.4)
                        edge_embedding += update
                    else:
                        count_edge = count_edge + 1
                        update = update_embedding(w_ckg, mid, n, 0.4)
                        edge_embedding += update
                if data['n_labels'][aa][0] == 3:
                    mid2 = data['mid_nodes'][aa][0]
                    mid1_id = np.where(data['nodes'] == mid2)
                    mid1 = data['mid_nodes'][mid1_id][0][0]
                    node_update_b = embedding_index[(entities[int(n)].lower()).replace(' ', '_')] * w_ckg[
                        id[1], mid1] * ((7 - np.sum(data['n_labels'][aa])) / 6) * w_ckg[mid1, mid2] * w_ckg[n, mid2]
                    if mid1 not in node_f:
                        count_edge = count_edge + 3
                        update = update_embedding(w_ckg, mid1, mid2, 0.4) + update_embedding(w_ckg, id[1], mid1, 0.8) \
                                 + update_embedding(w_ckg, mid2, n, 0.2)
                        edge_embedding += update
                    elif mid2 not in node_f:
                        count_edge = count_edge + 2
                        update = update_embedding(w_ckg, mid1, mid2, 0.4) + update_embedding(w_ckg, mid2, n, 0.2)
                        edge_embedding += update
                    else:
                        count_edge = count_edge + 1
                        update = update_embedding(w_ckg, mid2, n, 0.2)
                        edge_embedding += update
                if data['n_labels'][aa][1] == 3:
                    mid2 = data['mid_nodes'][aa][1]
                    mid1_id = np.where(data['nodes'] == mid2)
                    mid1 = data['mid_nodes'][mid1_id][0][1]
                    node_update_b = embedding_index[(entities[int(n)].lower()).replace(' ', '_')] * w_ckg[
                        id[0], mid1] * ((7 - np.sum(data['n_labels'][aa])) / 6) * w_ckg[mid1, mid2] * w_ckg[n, mid2]
                    if mid1 not in node_f:
                        count_edge = count_edge + 3
                        update = update_embedding(w_ckg, mid1, mid2, 0.4) + update_embedding(w_ckg, id[0], mid1, 0.8) \
                                 + update_embedding(w_ckg, mid2, n, 0.2)
                        edge_embedding += update
                    elif mid2 not in node_f:
                        count_edge = count_edge + 2
                        update = update_embedding(w_ckg, mid1, mid2, 0.4) + update_embedding(w_ckg, mid2, n, 0.2)
                        edge_embedding += update
                    else:
                        count_edge = count_edge + 1
                        update = update_embedding(w_ckg, mid2, n, 0.2)
                        edge_embedding += update
                if data['n_labels'][aa][0] == 1:
                    node_update_f = embedding_index[(entities[int(n)].lower()).replace(' ', '_')] * \
                                    ((7 - np.sum(data['n_labels'][aa])) / 6)
                if data['n_labels'][aa][1] == 1:
                    node_update_b = embedding_index[(entities[int(n)].lower()).replace(' ', '_')] * \
                                    ((7 - np.sum(data['n_labels'][aa])) / 6)
                if 101 <= n <= 170:
                    if data['n_labels'][aa][0] == 1:
                        prob[n - 101] = w_ckg[int(n), int(id[1])]
                    if data['n_labels'][aa][0] == 2:
                        mid = data['mid_nodes'][aa][0]
                        prob[n - 101] = w_ckg[int(mid), int(id[1])] * w_ckg[int(mid), int(n)]
                    if data['n_labels'][aa][0] == 3:
                        mid2 = data['mid_nodes'][aa][0]
                        mid1_id = np.where(data['nodes'] == mid2)
                        mid1 = data['mid_nodes'][mid1_id][0][0]
                        prob[n - 101] = w_ckg[int(mid1), int(id[1])] * w_ckg[int(mid1), int(mid2)] * w_ckg[int(mid2),
                        int(n)]
                    if data['n_labels'][aa][1] == 1:
                        prob[n - 101] = prob[n - 101] * w_ckg[int(n), int(id[0])]
                    if data['n_labels'][aa][1] == 2:
                        mid = data['mid_nodes'][aa][1]
                        prob[n - 101] = prob[n - 101] * w_ckg[int(mid), int(id[0])] * w_ckg[int(mid), int(n)]
                    if data['n_labels'][aa][1] == 3:
                        mid2 = data['mid_nodes'][aa][1]
                        mid1_id = np.where(data['nodes'] == mid2)
                        mid1 = data['mid_nodes'][mid1_id][0][1]
                        prob[n - 101] = prob[n - 101] * w_ckg[int(mid1), int(id[0])] * w_ckg[int(mid1), int(mid2)] * \
                                        w_ckg[int(mid2), int(n)]
                else:
                    count_node = count_node + 1
                    node_update_f_fin += node_update_f
                    node_update_b_fin += node_update_b
                if (id[0], id[1]) in w_ckg.keys():
                    prob = np.add(prob, w_ckg[(id[0], id[1])])
            relationship_embedding += edge_embedding
            p_ckg.append(prob)
            train_rel_embedding.append(relationship_embedding)
            embedding_index[(types[int(id[0])].lower()).replace(' ', '_')] += node_update_f_fin / count_node
            embedding_index[(types[int(id[0])].lower()).replace(' ', '_')] = embedding_index[
                                                                                 (types[int(id[0])].lower()).replace(
                                                                                     ' ', '_')] / 2
            embedding_index[(types[int(id[1])].lower()).replace(' ', '_')] += node_update_b_fin / count_node
            embedding_index[(types[int(id[1])].lower()).replace(' ', '_')] = embedding_index[
                                                                                 (types[int(id[1])].lower()).replace(
                                                                                     ' ', '_')] / 2
        else:
            p_ckg.append(prob)
            train_rel_embedding.append(relationship_embedding)
    return train_rel_embedding, p_ckg


def write2file(label):
    with open(r'ckg_' + label + '.txt', mode='w', newline='', encoding='utf8') as cf:
        wf = csv.writer(cf, delimiter=',')
        wf.writerow(['sub', 'obj', 'prob'])
        for word in list(embedding_index.keys()):  # 0~100
            wf.writerow([word, embedding_index[word]])
    cf.close()


adj_list, triplets, entity, relation2id, id2entity, id2relation, nei_node, w_ckg = process_files(
    file, saved_relation2id)

embedding_index = {}
entities = []
with open('numberbatch-en.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        if word in [i.lower().replace(' ', '_') for i in entity.keys()] + [j.split(' ')[0] for j in entity.keys()]:
            embedding = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = embedding

graphs = {'triplets': triplets, 'max_size': 1000000}
graphs['pos'], graphs['neg'] = sample_neg(adj_list, graphs['triplets'], 1,
                                          max_size=graphs['max_size'],
                                          constrained_neg_prob=0)
graphs['entity2relation'] = []
for sub in (set(entity.keys()) - set(predicates)):
    for relation in predicates:
        graphs['entity2relation'].append([entity[sub], entity[relation], 8])

graphs['sub2obj'] = []
for sub in types:
    for obj in types:
        graphs['sub2obj'].append([entity[sub], entity[obj], 8])

datum = links2subgraphs(adj_list, graphs, nei_node, entity, max_label_value)

CP = tf.placeholder(tf.float32, 70)
prob = tf.nn.softmax(CP)
