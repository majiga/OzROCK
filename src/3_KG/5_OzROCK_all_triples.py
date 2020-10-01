#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 10:33:35 2020

@author: majiga
"""

import os
import glob, json
import pandas as pd
import numpy as np
import csv
import time
import networkx as nx
#import matplotlib.pyplot as plt 
from graph_from_file import text2graph

FILENAME = r'Files/OzROCK-master/All_dataset.txt'
print("Data Files: ", FILENAME)

df = pd.read_csv(FILENAME,
                 delimiter=" ",
                 header=None,
                 quoting=csv.QUOTE_NONE,
                 encoding='latin1',
                 skip_blank_lines=True)
df = df.replace(np.nan, '', regex=True)
#print(df.head(3))
print('df_expert = ', len(df))

sentences = ' '.join(df[0].tolist()).split(' . ')

sentences = [s+' .' for s in sentences]
sentences = [' '.join(s.split()) for s in sentences]

print('Num sentences = ', len(sentences))

count_error = 0
count = 0

start_time = time.time()
FILE_TRIPLES = open(r'Files/OzROCK-master/All_dataset_triples.tsv', mode='w', newline='', encoding='utf-8')
tsv_writer = csv.writer(FILE_TRIPLES, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)

for s in sentences:
    try:
        graph = text2graph(s)
        #if graph.number_of_nodes() < 2:
        #    print('Less than 2 entities: ' + s)
        #    continue

        """
        #print("\n\nGRAPH:\n", nx.info(graph))
        print("\n\nNODES:\n")
        for n in graph.nodes(data=True):
            print(n)
        print("\n\nEDGES:\n")
        for s,t,a in graph.edges(data=True):
            print([s], [t], ' --- ', a['label'])
        """
        for s,t,a in graph.edges(data=True):
            #print([s], [t], ' --- ', a['label'])
            tsv_writer.writerow([s, t, a['label'], graph.node[s]['group'], graph.node[t]['group']])
        count += 1
        
        if count % 100 == 0:
            print('sentence number: ', count)
        #if count == 10:
        #    break

    except Exception as ex:
        print("Error: " + s + ", error msg: " + str(ex))
        count_error += 1
        #break
       
print("Number of files that had errors: ", count_error)

FILE_TRIPLES.close()

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
"""

elapsed_time = time.time() - start_time
print(str(len(sentences)), " duration = ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


"""
with open('wamex_files.csv', mode='w', newline='', encoding='utf-8') as file:
    csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #csv_writer.writerow(['head', 'relation', 'tail', head_group, tail_group])
    csv_writer.writerows(files_ids)
print("wamex_files.csv file is created.")
"""
