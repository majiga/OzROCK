#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 19:49:34 2020

@author: majiga

Build a Knowledge Graph from json graph files.
"""
import glob, os
import time, csv
import networkx as nx
import simplejson as json

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from networkx.drawing.nx_pydot import read_dot
import seaborn as sns
sns.set(style="white")

count_files = 0
count_files_error = 0

COMM_ = 'gold_deposit' # 'gold_deposit' or 'iron_ore'
FOLDER_GRAPHS = r"Files/10_gold_deposit_graphs/"
FOLDER_KG = r"Files/"

files = glob.glob(os.path.join(FOLDER_GRAPHS, '*.json'))
print("Files = ", len(files))

KG = nx.DiGraph(name = "KnowledgeGraph")


def save_graph(graph, fname):
    nodes = [{'id': n, 'group': graph.node[n]['group'], 'degree': str(graph.node[n]['degree'])} for n in graph.nodes()]
    links = [{'source': u, 'target': v, 'label': d['label']} for u, v, d in graph.edges(data=True)]
    with open(fname, 'w') as f:
        json.dump({'nodes': nodes, 'links': links}, f, indent=4,)            
    
def load_graph(filename):
    d = json.load(open(filename))
    #print(d, '\n\n')
    g = nx.DiGraph()    
    
    for n in d['nodes']:
        if n['group'] != 'OTHER':
            g.add_node(n['id'], group = n['group'])
        
    for n in d['links']:
        #print(n['source'], n['target'], n['label'])
        g.add_edge(n['source'], n['target'], label = n['label'])
    """
    print("\nNodes\n")
    for n in g.nodes(True):
        print(n)
    print("\nEdges\n")
    for a, b, c in g.edges(data=True):
        print(a, ' - ', b, ' - ', c)
    """
    return g

def join_graphs(big_g, small_g):
    G = nx.DiGraph()
    
    for n,d in big_g.nodes(data=True):
        if not G.has_node(n):  
            G.add_node(n, group = d['group'])
    for n,d in small_g.nodes(data=True):
        if not G.has_node(n):  
            G.add_node(n, group = d['group'])

    all_edges = {} # key=[s,t]; value=list of relations
    for s, t, d in big_g.edges(data=True):
        all_edges[s+';'+t] = d['label'].split(',')
    for s, t, d in small_g.edges(data=True):
        if s+';'+t in all_edges.keys():
            all_edges[s+';'+t] = all_edges[s+';'+t] + d['label'].split(',')
            #print('\n', all_edges[s+';'+t] + d['label'].split(','))            
        else:
            all_edges[s+';'+t] = d['label'].split(',')
        
    for key,value in all_edges.items():
        #print([key], [value])
        s, t = key.split(';')
        labels = list(set(value))
        labels = [l.strip() for l in labels]
        labels = [l for l in labels if l]
        G.add_edge(s, t, label=','.join(labels))
            
    """
    print("\nJoined KG:\n", nx.info(U))
    #print("\n\nNODES:\n")
    #for n in U.nodes(data=True):
    #    print(n)
    print("\n\nEDGES:\n")
    for s,t,a in U.edges(data=True):
        print([s], [t], '  ---  ', [a])
    """
    return G


for filename in files:
    try:
        start_time = time.time()
        print('\n', str(count_files) + " = File name: " + filename)


        graph = load_graph(filename)
        #print("Loaded a graph from a json ... \n", nx.info(graph), '\nLength of a graph = ', len(graph), '\n')             
        
        if len(graph) < 2:
            continue
        """
        # Save as csv of edges [source, target, label]
        with open(filename.rstrip('.json') + '.tsv', mode='w', newline='', encoding='utf-8') as file:
            csv_writer = csv.writer(file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for s,t,a in graph.edges(data=True):
                #csv_writer.writerow(['head', 'relation', 'tail', head_group, tail_group])
                labels = set(list(l for l in a['label'].split(',') if l))
                print('\nLabels:\n', labels)
                csv_writer.writerow([s,t,','.join(labels), graph.node[s]['group'], graph.node[t]['group']])            
        print("tsv file is created.") 
        """
        """
        print("\n\nNODES:\n")
        for n in graph.nodes(data=True):
            print(n)
        print("\n\nEDGES:\n")
        for s,t,a in graph.edges(data=True):
            print([s], [t], [a])
        """
        #KG = nx.compose(KG, graph) # compose do not keep all attributes
        KG = join_graphs(KG, graph)
                
        elapsed_time = time.time() - start_time
        print(str(count_files), " duration = ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        count_files += 1
        #break

    except Exception as ex:
        print("Error in file: " + filename + ", error msg: " + str(ex))
        count_files_error += 1
        #break
       
print("\nNumber of successful files: ", count_files)
print("Number of files that had errors: ", count_files_error)


def resolve_plurals(g):
    
    remove_nodes = {}
    add_edges = []
    print('\nResolve plurals :\n', nx.info(g))
    
    # deal with nodes
    for n, data in g.nodes(data=True):
        if n.endswith('s') and n.rstrip('s') in g.nodes():
            existing_node = n.rstrip('s')
            remove_nodes[n] = True            
            for s,t in list(g.in_edges(n)): # target node is n
                add_edges.append([s, existing_node, g[s][t]['label']])
            for s,t in list(g.out_edges(n)): # source node is n
                add_edges.append([existing_node, t, g[s][t]['label']])
    print('\n', add_edges)
            
    for s, t, label in add_edges:
        #print('- add egde: ', [s], [t], [label])        
        g.add_edge(s, t, label=label)
    
    for n in remove_nodes: # remove the merged nodes
        if n in g.nodes():
            #print('--- remove node: ', n)
            g.remove_node(n)
    
    print('Resolved plurals after:\n', nx.info(g))
    return g


print(nx.info(KG))
KG = resolve_plurals(KG)
print(nx.info(KG))

"""
print("\n\nEDGES:\n")
for s,t,a in KG.edges(data=True):
    print([s], [t], '  ---  ', [a])
"""


# Save graph files
#nx.write_gpickle(KG, FOLDER_KG + "KG_gold_deposit.gpickle")
nx.write_gpickle(KG, FOLDER_KG + "KG_" + COMM_ + ".gpickle")
#G = nx.read_gpickle("KG.gpickle")

# Add degree to the KG
degree_dict = dict(KG.degree(KG.nodes()))
nx.set_node_attributes(KG, degree_dict, 'degree')

#save_graph(KG, FOLDER_KG + "KG_gold_deposit.json")
save_graph(KG, FOLDER_KG + "KG_" + COMM_ + ".json")


def draw(G, measures, measure_name):
    plt.figure(figsize=(15,10))
    pos = nx.spring_layout(G)
    #nx.draw(G, pos, with_labels=True)
    nodes = nx.draw_networkx_nodes(G, pos, node_size=250, cmap=plt.cm.plasma, 
                                   node_color=list(measures.values()),
                                   nodelist=measures.keys())
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1))    
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos)

    plt.title(measure_name)
    plt.colorbar(nodes)
    #plt.axis('off')
    plt.show()


"""
What are the first 10 key words come to your mind, when you think of 'gold'?
Deposit
Sulfides
Veins
Alteration
Faults
Shear zones
Orogenic
Fluids
Exploration
Timing
"""

if ('gold deposit' in KG.nodes()):
    ego = nx.ego_graph(KG, 'gold deposit', radius=1)
    #nx.write_gpickle(ego, FOLDER_KG + "gold_deposit.gpickle")
    save_graph(ego, FOLDER_KG + "ego_" + COMM_ + ".json")
    print("Subgraph for gold deposit: ", nx.info(ego))

if ('nickel' in KG.nodes()):
    ego = nx.ego_graph(KG, 'nickel', radius=1)
    #nx.write_gpickle(ego, FOLDER_KG + "gold_deposit.gpickle")
    save_graph(ego, FOLDER_KG + "ego_nickel.json")
    draw(ego, nx.degree_centrality(ego), 'Degree Centrality')
    print("Subgraph for nickel: ", nx.info(ego))


draw(ego, nx.degree_centrality(ego), 'Degree Centrality')


sorted(ego.degree, key=lambda x: x[1], reverse=True)
#edges=sorted(ego.edges(data=True), key=lambda t: t[2].get('degree', 1))

for n in sorted(ego.degree, key=lambda x: x[1], reverse=True):
    print(n)
    
# Top 20 terms for iron ore
H = KG.subgraph([ 'gold', 'gold deposit', 'coolgardie', 'sulphide', 'archaean', 'diorite',
                 'goldfields', 'lode gold deposit', 'porphyry', 'amphibolite', 'kalgoorlie',
                 'supergene', 'yilgarn craton', 'quartz', 'amphibolite facies', 'quartz vein'])
print(list(H.edges(data=True)))
draw(H, nx.degree_centrality(H), 'Degree Centrality')
save_graph(H, FOLDER_KG + "top_" + COMM_ + ".json")


#plt.figure(figsize=(15,15))
#nx.draw(ego, with_labels=True)

"""
for n in KG.nodes(data=True):
    print(n)
for s,t,a in ego.edges(data=True):
    print([s], [t], [a])
"""
        

# KG to patterns

edges_ranking = {} # count frequency
edges_relations = {} # list of relation defining words
for s,t,a in KG.edges(data=True):    
    if KG.node[s]['group'] == KG.node[t]['group']:
        continue
    key = KG.node[s]['group']+'\t'+KG.node[t]['group']    
    if key in edges_ranking.keys():
        edges_ranking[key] = edges_ranking[key] + 1
        edges_relations[key] = edges_relations[key] + a['label'].split(',')
    else:
        edges_ranking[key] = 1
        edges_relations[key] = a['label'].split(',')
edges_ranking_sorted = sorted(edges_ranking.items(), key=lambda x: x[1], reverse=True)
for k in edges_ranking_sorted:
    print(k)

for k,v in edges_relations.items():
    rels = set([x for x in v if x])
    if rels:
        print(k, rels)
        
        
