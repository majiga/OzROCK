# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:55:31 2019

@author: Majigsuren Enkhsaikhan
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


G = nx.DiGraph()

def create_strats_graph():
    columns = ['Stratno', 'Stratigraphic Name', 'Rank', 'Previous Name', 'Previous Name Stratno',
               'Parent Stratno', 'Parent Name',
               'Top Minimum Age Name', 'Base Maximum Age Name',
               'Primary Lithology Group']
    #df = pd.read_csv(r"C:/Users/20230326/wamex/data/WAStrat/WA_Stratigraphic_Names.csv",
    df = pd.read_csv(r"WA_Stratigraphic_Names.csv",
                     encoding='latin-1', sep='\s+,\s+', delimiter=',',
                     usecols=columns, skipinitialspace=True, dtype=str)
    #df = df.fillna('0')
    #print(df.head())
    #print(df.columns)
    
    # Nodes
    for index, row in df.iterrows():
        if pd.isna(row['Stratno']) or pd.isna(row['Stratigraphic Name']) or not row['Stratno'].isdigit():
            continue
        node_name = row['Stratigraphic Name'].lower().strip()
        if (node_name not in G.nodes()):
            G.add_node(node_name, index=row['Stratno'],
                       #id_old=row['Previous Name Stratno'], name_old=row['Previous Name'],
                       rank=row['Rank'],
                       min_age=row['Top Minimum Age Name'], max_age=['Base Maximum Age Name'],
                       lith=row['Primary Lithology Group'], current_name='1',
                       group='STRAT')
    #print(nx.info(G))
    
    for index, row in df.iterrows():
        if pd.isna(row['Parent Stratno']) or pd.isna(row['Parent Name']) or not row['Parent Stratno'].isdigit():
            continue
        node_name = row['Parent Name'].lower().strip()
        if (node_name not in G.nodes()):
            G.add_node(node_name, index=row['Parent Stratno'], current_name='1', group='STRAT')                       
    #print(nx.info(G))

    for index, row in df.iterrows():
        if pd.isna(row['Previous Name Stratno']) or pd.isna(row['Previous Name']) or not row['Previous Name Stratno'].isdigit():
            continue
        node_name = row['Previous Name'].lower().strip()
        if (node_name not in G.nodes()):
            G.add_node(node_name, index=row['Previous Name Stratno'], current_name='0', group='STRAT')
    #print(nx.info(G))
    
    # Edges
    for index, row in df.iterrows():
        if (pd.isna(row['Parent Stratno']) or pd.isna(row['Stratno']) or not row['Stratno'].isdigit() or not row['Parent Stratno'].isdigit()):
            continue
        node_name1 = row['Parent Name'].lower().strip()
        if (node_name not in G.nodes()):
            G.add_node(node_name, group='STRAT')
        node_name2 = row['Stratigraphic Name'].lower().strip()
        if (node_name not in G.nodes()):
            G.add_node(node_name, group='STRAT')
        G.add_edge(node_name1, node_name2, label='parent_of')
    #print(nx.info(G))
    
    for index, row in df.iterrows():
        if (pd.isna(row['Previous Name Stratno']) or pd.isna(row['Stratno']) or pd.isna(row['Previous Name']) or
            not row['Stratno'].isdigit() or not row['Previous Name Stratno'].isdigit()):
            continue
        node_name1 = row['Stratigraphic Name'].lower().strip()
        if (node_name not in G.nodes()):
            G.add_node(node_name, group='STRAT')
        node_name2 = row['Previous Name'].lower().strip()
        if (node_name not in G.nodes()):
            G.add_node(node_name, group='STRAT')
        G.add_edge(node_name1, node_name2, label='current_of')
    #print(nx.info(G))
    
    
    #for n in G.nodes(data=True):
    #    print(n)
    #for s,t,a in G.edges(data=True):
    #    print([s], [t], [a['r']])

def get_stratigraphic_hierarchy(stratsList):
    
    create_strats_graph()
    
    # Create ego graph of a given node
    #ego = nx.ego_graph(G, '70159')
    #reachable_states = nx.descendants(G, '70159')
    #sub = G.subgraph(reachable_states)
    #print(nx.info(sub), '\n')
    additional_nodes = set()

    for node in stratsList:
        if G.has_node(node):
            additional_nodes.update([node])
            parent_nodes = nx.ancestors(G, node)   # same as predecessors        
            additional_nodes.update(parent_nodes)
            #child_nodes = nx.descendants(G, node) # child_to_parents 
            #additional_nodes.update(child_nodes)
        else:
            G.add_node(node, group='STRAT')
            additional_nodes.update([node])
            
    sub = G.subgraph(additional_nodes)
    
    """
    print(nx.info(sub), '\n')
    #for n, data in sub.nodes(True):
    #    print(n, data)
    print('\nSub graph:\n')
    for s, t, a in sub.edges(data=True):
        print(s, t, a)
        print(sub.node[s])
        print(sub.node[t])
    """
    return sub

def draw_graph(sub, node_list):    
    # Draw graph
    plt.figure(figsize=(10,10))
    pos = nx.spring_layout(sub)
    nx.draw(sub, pos, node_color='blue', alpha=0.6, node_size=300, with_labels=True)
    
    # Draw ego as large and red
    nx.draw_networkx_nodes(sub, pos, nodelist=node_list, node_size=500, node_color='r', alpha=0.6)
    
    #node_labels = nx.get_node_attributes(sub,'name')
    #nx.draw_networkx_labels(sub, pos, labels = node_labels)
    
    edge_labels = nx.get_edge_attributes(sub,'r')
    nx.draw_networkx_edge_labels(sub, pos, labels = edge_labels)
    plt.show()
    
    
if __name__ == "__main__":
    
    nodes = ['pincunah hill formation', # Cardinal Formation (previous name = pincunah hills formation)
             'Honeyeater Basalt'.lower(), # Honeyeater Basalt
             'Gorge Creek Formation'.lower(), # Gorge Creek Group (previous name = Gorge Creek Formation)
             'wooramel group',
             'corboy formation']
    sub = get_stratigraphic_hierarchy(nodes)
    
    print(nx.info(sub), '\n')
    for n, data in sub.nodes(True):
        print(n, data)
    
    draw_graph(sub, nodes)
