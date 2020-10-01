#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 11:42:47 2019

@author: majiga

# If you need to match large terminology lists, you can also use the PhraseMatcher and create Doc objects instead of token patterns, which is much more efficient overall. 
The Doc patterns can contain single or multiple tokens.

# Save all dictionary items in this pattern in the json file
patterns = [
                {"label": "NOUN_CHUNK", "pattern": [{"LOWER": "macleod"}, {"LOWER": "member"}]},
                {"label": "NOUN_CHUNK", "pattern": [{"LOWER": "iron"}]}
                ]
"""
import io, os, json

DICTIONARY_FOLDER = r"../0Domain_Vocabulary/"
#DICTIONARY_FOLDER = r"C:/Users/20230326/0Domain_Vocabulary/"


def read_vocabulary(file_name, term_type):
    terms = {}
    with open(file_name, 'r', encoding='latin1') as f:
        lines = f.readlines()
        for t in lines:
            term = t.lower().strip()
            if term:
                terms[term] = term_type
    print(term_type + ' terms: ', len(terms))
    return terms

# Read domain vocabulary
def read_domain_vocabulary():
    abspath = os.path.abspath('')
    os.chdir(abspath)

    rocks = read_vocabulary(DICTIONARY_FOLDER + "2019_rocks_mindat_gswa.txt", 'ROCK')
    rocks_plural = {r+'s':t for r,t in rocks.items()}
    minerals = read_vocabulary(DICTIONARY_FOLDER + "2019_minerals_mindat_gswa.txt", 'MINERAL')
    minerals_plural = {m+'s':t for m,t in minerals.items()}
    timescales = read_vocabulary(DICTIONARY_FOLDER + "2019_geological_eras.txt", 'TIMESCALE')
    ore_deps = read_vocabulary(DICTIONARY_FOLDER + "2019_ores_deposits.txt", 'ORE_DEPOSIT')
    ore_deps_plural = {o+'s':t for o,t in ore_deps.items()}
    strats = read_vocabulary(DICTIONARY_FOLDER + "2019_stratigraphy_gswa.txt", 'STRAT')
    strats_plural = {s+'s':t for s,t in strats.items()}
    locations = read_vocabulary(DICTIONARY_FOLDER + "2019_locations.txt", 'LOCATION')
    
    return {**rocks, **minerals, **timescales, **ore_deps, **strats, **locations,
            **rocks_plural, **minerals_plural, **ore_deps_plural, **strats_plural}


# read vocabulary
    
vocabulary = read_domain_vocabulary()

voc_dictionary = []
for k, v in vocabulary.items():
    words = []
    for word in k.split():
        w = {"LOWER": word}
        words.append(w)
    #term = {"label": "NOUN_CHUNK", "pattern": words}
    term = {"label": v, "pattern": words}
    voc_dictionary.append(term)
    
print("\nDictionary length = ", len(voc_dictionary))
    
# Write JSON file
with io.open('VOCABULARY_TYPED.json', 'w', encoding='utf8') as outfile:
    str_ = json.dumps(voc_dictionary,
                      indent=4,
                      sort_keys=True,
                      separators=(',', ': '),
                      ensure_ascii=False)
    outfile.write(str_)

print("VOCABULARY_TYPED.json file is created. ")

# Read JSON file
#with open('data.json') as data_file:
#    data_loaded = json.load(data_file)

