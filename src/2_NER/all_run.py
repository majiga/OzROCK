#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 09:19:24 2019

@author: majiga
"""
import glob, os, json
#import pandas as pd
#import numpy as np
#import csv
import time
import networkx as nx
import matplotlib.pyplot as plt
   
from graph_from_file import text2graph


count_files = 0
count_files_error = 0

#FOLDER_ALL_WAMEX = r"C:/Users/20230326/wamex/data/test_1/"
FOLDER_ALL_WAMEX = r"C:/Users/20230326/wamex/data/wamex_xml/"
FOLDER_WAMEX_graph = r"C:/Users/20230326/wamex/data/2020_wamex_aGraph_aFile_Feb27/"

FOLDER_TESTSET_SKIP = r"C:/Users/20230326/wamex/data/ANNOTATION_100_WAMEX/"


#FOLDER_WAMEX_annotation = r"C:/Users/20230326/wamex/data/2020_wamex_annotation/"

#FOLDER_ALL_WAMEX = r"C:/Users/20230326/wamex/data/wamex_xml/"
#FOLDER_ALL_WAMEX = r"/Users/majiga/Documents/wamex/data/test_iron_ore/"
#FOLDER_ALL_WAMEX_fix = r"/Users/majiga/Documents/wamex/data/test_iron_ore_graphs/"

#FOLDER_ALL_WAMEX = r"/Users/majiga/Documents/wamex/data/test_vms/"
#FOLDER_ALL_WAMEX_fix = r"/Users/majiga/Documents/wamex/data/test_vms_graphs/"

#FOLDER_ALL_WAMEX = r"/Users/majiga/Documents/wamex/data/test_gold/"
#FOLDER_ALL_WAMEX_fix = r"/Users/majiga/Documents/wamex/data/test_gold_graphs/"

files = glob.glob(os.path.join(FOLDER_ALL_WAMEX, '*.json'))
print("Files = ", len(files))

files_skip = glob.glob(os.path.join(FOLDER_TESTSET_SKIP, '*.txt'))

files_skip_names = [(str(filename).split('\\')[-1:][0]).split('.')[-2:][0] for filename in files_skip]

print("Files in the test set, so need to skip = ", len(files_skip))

def create_triples(g):
    triples = []
    ore_deps = []
    #print('Creating the triples from the graph\n')
    for s,t,a in g.edges(data=True):
        s_group = g.node[s]['group']
        t_group = g.node[t]['group']
        if (s_group == 'ORE_DEPOSIT' and s_group not in ore_deps):
            ore_deps.append(s)
        if (t_group == 'ORE_DEPOSIT' and t_group not in ore_deps):
            ore_deps.append(t)
        if a:
            triples.append([s, a['label'], t, g.node[s]['group'], g.node[t]['group']])
        else:
            triples.append([s, [], t, g.node[s]['group'], g.node[t]['group']])
    return triples, ore_deps


def create_triples_from_graph(g):
    triples = []
    #print('Creating the triples from the graph\n')
    for s,t,a in g.edges(data=True):
        s_group = g.node[s]['group']
        t_group = g.node[t]['group']
        if a:
            triples.append([s, a['label'], t, s_group, t_group])
        else:
            triples.append([s, [], t, s_group, t_group])
    return triples


KG = nx.DiGraph(name = "KnowledgeGraph")
files_ids = []

fh = open("annotation.txt","w")
fh.close()

count_sentences = 0

for filename in files:
    try:
        start_time = time.time()
        print('\n======= ', str(count_files) + " = File name: " + filename)

        #filename_extension = str(filename).split('/')[-1:][0] # for macbook
        filename_extension = str(filename).split('\\')[-1:][0] # for windows
        filename_new_graph_no_extension = FOLDER_WAMEX_graph + str(filename_extension).split('.')[-2:][0]
        filename_new_graph = filename_new_graph_no_extension + '.json'

        if filename_new_graph_no_extension in files_skip_names:
            print('File is in TEST DATASET, no need to process. Name: ' + filename_new_graph)
            continue

        if os.path.isfile(filename_new_graph): # If graph file is already exists, no need to process again
            print('File exists, no need to process. Name: ' + filename_new_graph)
            continue

        with open(filename, 'r') as f:
            data = f.read()
        if len(data.split()) < 5:
            print('File is too short. Name: ' + filename_new_graph)
            continue
        
        graph, annotation = text2graph(data)
        if graph.number_of_nodes() < 2 or len(annotation) < 3: 
            print('File is too short. Name: ' + filename_new_graph)
            continue

        """
        # Save annotation file
        print('Len annotation = ', len(annotation))
        fh = open("annotation.txt", "a")        
        
        annotated = False
        annotated_sentences = []
        aSent = []
        
        for word, tag in annotation:
            aSent.append([word, tag])
            if tag != 'O':
                #print(word, tag)
                annotated = True
            if word == '.':
                if annotated == True:
                    annotated_sentences = annotated_sentences + aSent
                    count_sentences += 1
                annotated = False
                aSent = []                            
        
        for word, tag in annotated_sentences:
            fh.write(word + ' ' + tag + '\n')
            if word == '.':
                fh.write('\n')
        fh.close()
        """

        #print("\n\nGRAPH:\n", nx.info(graph))
        """
        print("\n\nNODES:\n")
        for n in graph.nodes(data=True):
            print(n)
        print("\n\nEDGES:\n")
        for s,t,a in graph.edges(data=True):
            print([s], [t], ' --- ', a['label'])
        """
            
        # Add degree
        degree_dict = dict(graph.degree(graph.nodes()))
        nx.set_node_attributes(graph, degree_dict, 'degree')
    
        # Save graph files
        nodes = [{'id': n, 'group': graph.node[n]['group'], 'degree': graph.node[n]['degree']} for n in graph.nodes()]
        #links = [{'source': u, 'target': v, 'label': data['label'], 'data': data} for u, v, d in graph.edges(data=True)]
        links = [{'source': u, 'target': v, 'label': d['label']} for u, v, d in graph.edges(data=True)]
        with open(filename_new_graph, 'w') as f:
            json.dump({'nodes': nodes, 'links': links}, f, indent=4,)

        count_files += 1
        elapsed_time = time.time() - start_time
        print(str(count_files), " duration = ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        #print(count_files, filename_new)
        
        #if count_files == 3:
        #    break

    except Exception as ex:
        print("Error in file: " + filename + ", error msg: " + str(ex))
        count_files_error += 1
        #break
       
print("Number of annotated sentences: ", count_sentences)
print("Number of successful files: ", count_files)
print("Number of files that had errors: ", count_files_error)


"""
with open('wamex_files.csv', mode='w', newline='', encoding='utf-8') as file:
    csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #csv_writer.writerow(['head', 'relation', 'tail', head_group, tail_group])
    csv_writer.writerows(files_ids)
print("wamex_files.csv file is created.")
"""
