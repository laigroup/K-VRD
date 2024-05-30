from anytree import Node, RenderTree, RenderTree, LevelOrderGroupIter
import rdflib
import os
from rdflib import URIRef
import numpy as np

# from sets import Set

ontology_dir = "data/ontology"

predicates = np.genfromtxt(os.path.join(ontology_dir, "predicates.csv"), dtype="str", delimiter=",")


def aggregate_equiv(equiv_set, input_vec, predicate_dict, aggregator):
    for pair in equiv_set:
        aggregator_vec = []
        for pred in pair.split(","):
            aggregator_vec.append(input_vec[predicate_dict[pred]])
        if aggregator is "max":
            aggregator_value = np.max(aggregator_vec)
        elif aggregator is "min":
            aggregator_value = np.min(aggregator_vec)
        elif aggregator is "mean":
            aggregator_value = np.mean(aggregator_vec)

        for pred in pair.split(","):
            input_vec[predicate_dict[pred]] = aggregator_value

    return input_vec


def refine_equiv(values_of_predicates, selected_predicates, aggregator):
    equiv_set = []
    with open(os.path.join(ontology_dir, "equiv.csv"), 'r', encoding = "GB2312") as file:
        for k, line in enumerate(file):
            line = line.strip('\"')
            line = line.strip('\"\n')
            equiv_set.append(line)
    predicate_dict = {}
    for idx in range(len(selected_predicates)):
        predicate_dict[selected_predicates[idx].replace(" ", "_")] = idx
    for idx_pred in range(len(values_of_predicates)):
        values_of_predicates[idx_pred] = aggregate_equiv(equiv_set, values_of_predicates[idx_pred], predicate_dict, aggregator)

    return values_of_predicates
