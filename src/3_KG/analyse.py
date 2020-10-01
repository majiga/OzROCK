#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 13:40:06 2020

@author: majiga
"""
import glob, os, csv
import time
import networkx as nx
import simplejson as json

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from networkx.drawing.nx_pydot import read_dot
import seaborn as sns
sns.set(style="white")

"""
# Iron ore
#FILENAME = r'Files/semantic_graph_iron_ore.json'
FILENAME = r'Files/semantic_graph_iron_ore.json'
Graph1 = r'Files/iron_ore_subgraph1.json'
Graph2 = r'Files/iron_ore_subgraph2.json'
Graph3 = r'Files/iron_ore_subgraph3.json'
Graph4 = r'Files/iron_ore_subgraph4.json'
keyword = 'hematite'
"""
# Gold deposit
#FILENAME = r'Files/semantic_graph_gold_deposit.json'
FILENAME = r'Files/semantic_graph_gold_deposit.json'
Graph1 = r'Files/gold_deposit_subgraph1.json'
Graph2 = r'Files/gold_deposit_subgraph2.json'
Graph3 = r'Files/gold_deposit_subgraph3.json'
Graph4 = r'Files/gold_deposit_subgraph4.json'
keyword = 'gold'

def save_graph(graph, fname):
    nodes = [{'id': n, 'group': graph.nodes[n]['group'], 'degree': str(graph.nodes[n]['degree'])} for n in graph.nodes()]
    links = [{'source': u, 'target': v, 'label': d['label']} for u, v, d in graph.edges(data=True)]
    with open(fname, 'w') as f:
        json.dump({'nodes': nodes, 'links': links}, f, indent=4,)

def load_graph(filename):
    d = json.load(open(filename))
    g = nx.DiGraph()
    for n in d['nodes']:
        if n['group'] != 'OTHER':
            g.add_node(n['id'], group = n['group'], degree = n['degree'])
    for n in d['links']:
        g.add_edge(n['source'], n['target'], label = n['label'])
    return g

def resolve_plurals(g):    
    remove_nodes = {}
    add_edges = []
    #print('\nResolve plurals :\n', nx.info(g))
    
    # deal with nodes
    for n, data in g.nodes(data=True):
        if n.endswith('s') and n.rstrip('s') in g.nodes():
            existing_node = n.rstrip('s')
            remove_nodes[n] = True            
            for s,t in list(g.in_edges(n)): # target node is n
                add_edges.append([s, existing_node, g[s][t]['label']])
            for s,t in list(g.out_edges(n)): # source node is n
                add_edges.append([existing_node, t, g[s][t]['label']])
    #print('\n', add_edges)
            
    for s, t, label in add_edges:
        #print('- add egde: ', [s], [t], [label])        
        g.add_edge(s, t, label=label)
    
    for n in remove_nodes: # remove the merged nodes
        if n in g.nodes():
            #print('--- remove node: ', n)
            g.remove_node(n)
    
    #print('Resolved plurals after:\n', nx.info(g))
    return g

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

# Create a subgraph
def create_subgraph(G, categories, subgraph_filename):
    sub_graph = G.subgraph([n for n,d in G.nodes(data = True) if d['group'] in categories])
    draw(sub_graph, nx.degree_centrality(sub_graph), 'Degree Centrality')
    #degree_dict = dict(sub_graph.degree(sub_graph.nodes()))
    #nx.set_node_attributes(sub_graph, degree_dict, 'degree')
    save_graph(sub_graph, subgraph_filename)
    #print(G.nodes(True))


######### START ###########
def __main__():
    start_time = time.time()

    graph = load_graph(FILENAME)

    print(nx.info(graph))

    g = resolve_plurals(graph)
    #draw(g, nx.degree_centrality(g), 'Degree Centrality')


    # Ego graph
    if (keyword in g.nodes()):
        ego = nx.ego_graph(g, keyword, radius=1)
        #ego.remove_node('australia')
        #ego.remove_node('western australia')
        #ego.remove_node('diamond')
        #ego.remove_node('dolerite host rocks')
        if keyword == 'hematite':
            node_list = [n for n in ego.nodes()] + ['archaean', 'nammuldi member', 'hamersley group', 'mount newman member']            
        else:
            node_list = [n for n in ego.nodes()]
        print('\nEgo graph nodes: ', node_list)
        ego = g.subgraph(node_list)
        #sub_graph = g.subgraph(ego.nodes() + ['archaean', 'nammuldi member', 'hammersley group'])
        draw(ego, nx.degree_centrality(ego), 'Degree Centrality')
        #nx.write_gpickle(ego, FOLDER_KG + "iron_ore.gpickle")
                
        save_graph(ego, r"Files/ego_" + keyword + ".json")
        print("Ego for ", keyword, ': ' , nx.info(ego))

        """
        print('\nNODES:')
        for n in ego.nodes(data=True):
            print(n)
        print('\nEDGES:')
        for s,t,a in ego.edges(data=True):
            print([s], [t], [a])
        """
    
    create_subgraph(g, ['ROCK', 'STRAT', 'TIMESCALE'], Graph1)
    create_subgraph(g, ['ORE_DEPOSIT', 'MINERAL', 'LOCATION'], Graph2)
    create_subgraph(g, ['ORE_DEPOSIT', 'MINERAL'], Graph3)
    create_subgraph(g, ['LOCATION', 'STRAT'], Graph4)

    elapsed_time = time.time() - start_time
    print("DONE. duration = ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


    """
    #sub_minerals = g.subgraph([n for n,d in g.nodes(data = True) if d['group'] == 'MINERAL'])
    #draw(sub_minerals, nx.degree_centrality(sub_minerals), 'Degree Centrality')

    sub_locs = g.subgraph([n for n,d in g.nodes(data = True) if d['group'] == 'LOCATION'])
    draw(sub_locs, nx.degree_centrality(sub_locs), 'Degree Centrality')

    sub_ods = g.subgraph([n for n,d in g.nodes(data = True) if d['group'] == 'ORE_DEPOSIT' or d['group'] == 'MINERAL'])
    draw(sub_ods, nx.degree_centrality(sub_ods), 'Degree Centrality')

    sub_tss = g.subgraph([n for n,d in g.nodes(data = True) if (d['group'] == 'TIMESCALE' or d['group'] == 'STRAT' or d['group'] == 'ROCK')])
    draw(sub_tss, nx.degree_centrality(sub_tss), 'Degree Centrality')

    #sub_strats = g.subgraph([n for n,d in g.nodes(data = True) if d['group'] == 'STRAT'])
    #draw(sub_strats, nx.degree_centrality(sub_strats), 'Degree Centrality')
    """
