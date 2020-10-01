#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 09:19:24 2019

@author: majiga
"""
import os, sys
import glob, json
#import pandas as pd
#import numpy as np
#import csv
import time, datetime
import csv
import networkx as nx
import simplejson as json
import matplotlib.pyplot as plt

from graph_from_file import text2graph
from analyse import load_graph, resolve_plurals
"""
import logging
# debug(), info(), warning(), error(), and critical(), basicConfig() 
logging.basicConfig(format="%(asctime)s: [%(levelname)s], %(message)s", 
                    filemode='w',
                    filename="run.log",
                    level=logging.DEBUG) #, stream=sys.stdout)
#log=logging.getLogger(__name__)
#logging.basicConfig(filename='1_Run_Folder.log',
#                    filemode='a',
#                    format='%(asctime)s %(message)s', # '%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', # '%(message)s'
#                    datefmt = '%H:%M:%S', # '%m/%d/%Y %I:%M:%S %p', #datefmt='%H:%M:%S',
#                    level=logging.DEBUG)
"""
start_all_time = time.time()
print('STARTED: ', datetime.datetime.now())


#FOLDER_ALL_WAMEX = r"C:/Users/20230326/wamex/data/test_1/"

#FOLDER_FILES = r"Files/test/"
#FOLDER_FILES = r"Files/10_iron_ore/"
#FOLDER_GRAPHS = r"Files/10_iron_ore_graphs/"

#FOLDER_FILES = r"Files/10_gold_deposit/"
#FOLDER_GRAPHS = r"Files/10_gold_deposit_graphs/"

FOLDER_FILES = r"../wamex_data/wamex_xml/"
#FOLDER_GRAPHS = r"../wamex_data/wamex_graphs/"
#FOLDER_GRAPHS = r"../wamex_data/wamex_graphs_fast/"
FOLDER_GRAPHS = r"../wamex_data/wamex_graphs_fastest/"

# Read semantic relations from in a symantic_relations.csv file
semantic_relations_dict = {}
with open('Files/symantic_relations.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    next(csv_reader, None)
    for row in csv_reader:
        r_list = row[3:]
        rels = [r.strip().split(' ')[0].lower() for r in r_list if r]
        semantic_relations_dict[row[1]] = rels
print("KG relation types are loaded from symantic_relations.csv file.")
#print("KG relation types to identify:\n")
#for k, v in semantic_relations_dict.items():
#    print(k + ': ' + str(v))

def save_graph(graph, fname):
    nodes = [{'id': n, 'group': graph.nodes[n]['group'], 'degree': str(graph.nodes[n]['degree'])} for n in graph.nodes()]
    links = [{'source': u, 'target': v, 'label': d['label']} for u, v, d in graph.edges(data=True)]
    with open(fname, 'w') as f:
        json.dump({'nodes': nodes, 'links': links}, f, indent=4,)
    #print('Graph file %s is created.', fname)
"""
def load_graph(filename):
    d = json.load(open(filename))
    g = nx.DiGraph()
    for n in d['nodes']:
        if n['group'] != 'OTHER':
            g.add_node(n['id'], group = n['group'], degree = n['degree'])
    for n in d['links']:
        g.add_edge(n['source'], n['target'], label = n['label'])
    return g
"""

def reverse_direction(s, r, t):
    if r == 'overlies':
        return t, 'overlain_by', s
    if r == 'intrude':
        return t, 'intruded_by', s
    if r == 'hosts':
        return t, 'hosted_in', s
    if r == 'dominated':
        return t, 'dominated_by', s
    return s, r, t

##############################################################################
# Cleaning relations data
def update_dict(d, key):
    if key in d.keys():
        d[key] += 1
    else:
        d[key] = 1
    return d

def clean_relations_in_triples(input_triples):
    #print('Num input triples = ', len(input_triples))    
    triples = []
    relations_only = {}
    for s, rel_list, t in input_triples:
        #print('triples: ', s, rel_list, t)
        source = s.lower().replace(' ', '_')
        target = t.lower().replace(' ', '_')
        #print('- ', rel_list)
        relations = [r.lower().replace(' ', '_') for r in rel_list]
        #print('-- ', relations)
        relations = [r.replace('within', 'in') for r in relations]
        relations = [r.replace('is_parent_of', 'contains') for r in relations]
        relations = [r.replace('is_current_of', 'current_name_of') for r in relations]  
        relations = [r.replace('could_', '') for r in relations]
        relations = [r.replace('occurs_in', 'occur_in') for r in relations]
        relations = [r.replace('consists_of', 'consist_of') for r in relations]
        relations = [r.replace('hosted_by', 'hosted_in') for r in relations]
        relations = [r.replace('with_interbedded', 'interbedded_with') for r in relations]
        relations = [r.replace('may_', '') for r in relations]
        relations = [r.replace('can_', '') for r in relations]
        #print('-- ', relations)
        for r in relations:
            #print('---- ', r)
            if not r or r'(' in r or r')' in r or 'not' in r:
                triples.append([source, '', target])
                continue
            elif any(char.isdigit() for char in r):
                triples.append([source, '', target])
                continue
            elif '.' in r or '-' in r:
                triples.append([source, '', target])
                continue
            elif r in ['in', 'at', 'within', 'on']:
                triples.append([source, 'in', target]) # source, target, relation
                update_dict(relations_only, 'in')
                continue
            elif len(r) > 1 and len(r) < 25:
                #rel = get_base_form(r) # lemmas # 1165 relations
                triples.append([source, r, target]) # source, target, relation
                update_dict(relations_only, r)
            else:
                triples.append([source, '', target])
    #print('[clean_relations_in_triples] Num Output Triples = ', len(triples))
    #print('[clean_relations_in_triples] Num Relations = ', len(relations_only))
    return triples


# Get the graph with word-levelrelations and return graph with classified relations
def define_relation_types(input_graph):
    g = resolve_plurals(input_graph)
    triples = []
    for s, t, d in g.edges(data=True):
        triples.append([s, d['label'].split(','), t])
    #print('\nNum triples in a file = ', len(triples))
    
    triples_clean = clean_relations_in_triples(triples)
    
    semantic_triples = []    
    for s,r,t in triples_clean: # relations are single strings here
        found = False
        for k,v in semantic_relations_dict.items():
            if r in v:
                if [s, r, t] not in semantic_triples:
                    semantic_triples.append([s, k, t])
                    #print('+ ', s, r, t)
                found = True
                continue
        if not found:
            if [s, '', t] not in semantic_triples:
                semantic_triples.append([s, '', t])
            #print('- relation not found: ', s, r, t)
    #print('\nLen cleaned triples = ', len(semantic_triples))
    #print(*semantic_triples, sep='\n')
    
    pairs_relations = {}
    def update_relations_for_pair(dct, s, r, t): # This takes relations into a list per [s t] pair
        key = s + '\t' + t
        if key in dct.keys():
            rels = dct[key]
            if r not in rels:
                dct[key].append(r)           
        else:
            dct[key] = [r]
    
    for s, r, t in semantic_triples: # relations are single strings here too
        if r in ['overlies', 'intrude', 'hosts', 'dominated']:
            new_s, new_r, new_t = reverse_direction(s, r, t)
            update_relations_for_pair(pairs_relations, new_s, new_r, new_t)
        else:
            update_relations_for_pair(pairs_relations, s, r, t)
    #print('\nRelation pairs dictionary: ', len(pairs_relations))
    #for k,v in pairs_relations.items(): print(k, v)
    
    Semantic_G = nx.DiGraph() 
    for k,v in pairs_relations.items():
        #print(k)
        s, t = k.split('\t')
        source = s.replace('_', ' ')
        target = t.replace('_', ' ')
        Semantic_G.add_node(source, group = g.nodes[source]['group'])
        Semantic_G.add_node(target, group = g.nodes[target]['group'])
        rels = [r for r in v if r]    
        if rels:
            Semantic_G.add_edge(source, target, label = ';'.join(rels))
        else:
            Semantic_G.add_edge(source, target, label = '')
    #print('\nSemantic graph: ', nx.info(Semantic_G))
    
    # Add degree
    degree_dict = dict(Semantic_G.degree(Semantic_G.nodes()))
    nx.set_node_attributes(Semantic_G, degree_dict, 'degree')
    
    return Semantic_G


##############################################################################    
# Load all file and run
files = glob.glob(os.path.join(FOLDER_FILES, '*.json'))
print("Total number of input files # = ", len(files))

skip_files = glob.glob(os.path.join(FOLDER_GRAPHS, '*.json'))
skip_files = [str(fn).split('/')[-1:][0] for fn in skip_files]
print("Total number of input files # = ", len(skip_files))


count_files = 0
count_files_error = 0
count_sentences = 0

for filename in files:
    try:
        start_time = time.time()
        filename_extension = str(filename).split('/')[-1:][0] # for macbook
        #filename_extension = str(filename).split('\\')[-1:][0] # for windows
        #filename_new_graph_no_extension = FOLDER_GRAPHS + str(filename_extension).split('.')[-2:][0]
        #filename_new_graph = filename_new_graph_no_extension + '.json'
        
        if filename_extension in skip_files:
            continue
        
        print('\n======= ', str(count_files) + " = File name: " + filename)

        """
        if os.path.isfile(filename_new_graph): # If graph file is already exists, no need to process again
            print('File exists, no need to process. Name: ' + filename_new_graph)
            continue
        """
        data = ''
        with open(filename, 'r') as f:
            data = f.read()
        if len(data.split()) < 5:
            print('File is too short. Name: ' + filename_extension)
            #logging.info('File is too short. Name: ' + filename_new_graph)
            continue
        
        graph = text2graph(data)
        if graph.number_of_nodes() < 2:
            print('Not enough entities. Name: ' + filename_extension)
            #logging.info('Not enough entities. Name: ' + filename_new_graph)
            continue

        # Classify the word-level relations into 14 relation types
        graph = define_relation_types(graph)
        
        # Save graph files
        save_graph(graph, FOLDER_GRAPHS + filename_extension)
        
        print("Duration = ", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)),
              '; Number of nodes = ', graph.number_of_nodes(),
              '; Number of edges = ', graph.number_of_edges())
        #logging.info("Duration = ", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)),
        #      '; Number of nodes = ', graph.number_of_nodes(),
        #      '; Number of edges = ', graph.number_of_edges())
        count_files += 1
        #break

    except Exception as ex:
        #logging.error('Exception: {ex}, filename: %s', filename + ", error msg: " + str(ex))
        print("Error in file: " + filename_extension + ", error msg: " + str(ex))
        count_files_error += 1
        #break
       
print("Number of successful files: ", count_files)
print("Number of files that had errors: ", count_files_error)
#logging.debug("Number of successful files: " + str(count_files))
#logging.debug("Number of files that had errors: " + str(count_files_error))
hours, rem = divmod(time.time() - start_all_time, 3600)
minutes, seconds = divmod(rem, 60)
print("Total processing time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
print('FINISHED: ', datetime.datetime.now())


"""
with open('wamex_files.csv', mode='w', newline='', encoding='utf-8') as file:
    csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #csv_writer.writerow(['head', 'relation', 'tail', head_group, tail_group])
    csv_writer.writerows(files_ids)
print("wamex_files.csv file is created.")
"""
"""
#print("\n\nGRAPH:\n", nx.info(graph))
print("\n\nNODES:\n")
for n in graph.nodes(data=True):
    print(n)
print("\n\nEDGES:\n")
for s,t,a in graph.edges(data=True):
    print([s], [t], ' --- ', a['label'])
"""
