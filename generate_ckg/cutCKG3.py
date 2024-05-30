import pandas as pd
import json
import numpy as np
import os
import nltk
from nltk.corpus import wordnet
from change_KG import *

label = 'CKG_and'
ontology_dir = "data/ontology"

types = np.genfromtxt(os.path.join(ontology_dir, "classes.csv"), dtype="str", delimiter=",")
predicates = np.genfromtxt(os.path.join(ontology_dir, "predicates.csv"), dtype="str", delimiter=",")

for p in predicates:
    p = str(p).replace(' ', '_')

synonym_type = {}
synonym_pred = {}

entity = types.tolist()
relation = predicates.tolist()


class Sysnom_n:

    def __init__(self, word):
        self.word = word
        self.sys = self.get_sys()

    def get_sys(self):
        synonym = {}
        for sys in wordnet.synsets(self.word, pos=wordnet.NOUN):
            if sys.name().split('.')[0] not in synonym.keys():
                synonym[sys.name().split('.')[0]] = 0
            count = 0
            for cc in sys.lemmas():
                count = count + cc.count()
            synonym[sys.name().split('.')[0]] = synonym[sys.name().split('.')[0]] + count
        if self.word in synonym.keys():
            del synonym[self.word]
        synonym = {key: value for key, value in synonym.items() if value > 0}
        all = np.sum(list(synonym.values()))
        for ii in synonym.keys():
            synonym[ii] = synonym[ii] / all
        return synonym


for aa in types:
    synonym_type[aa] = Sysnom_n(aa)
    entity += list(synonym_type[aa].sys.keys())
node = entity + relation

FILE = 'concept_EN.csv'
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', 100000)

data = pd.read_csv(FILE, encoding='ISO-8859-1')
# data = data[~data['uri'].str.contains('[\u0020-\u002E]')]

data = data[~data['uri'].str.contains('[\u0030-\u0040]')]
fd = pd.DataFrame(None, columns=['uri', 'relation', 'start', 'end', 'weights'])

for index, row in data.iterrows():
    if (row['start'].split('/')[3] in node) & (row['end'].split('/')[3] in node):
        fd = fd.append(row, ignore_index=True)

fd.to_csv(label + ".csv")

changeCKG(label, synonym_type)
