#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 13:36:38 2020

@author: majiga
"""
import os
import csv
import pandas as pd
import re
import json
import spacy
from spacy import displacy
from spacy.pipeline import EntityRuler
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span, Token
from spacy.attrs import intify_attrs
nlp = spacy.load("en_core_web_sm")
#import neuralcoref
import networkx as nx
import matplotlib.pyplot as plt

from stratigraphy_hierarchy import get_stratigraphic_hierarchy

abspath = os.path.abspath('') ## String which contains absolute path to the script file
#print(abspath)
os.chdir(abspath)

DICTIONARY_FOLDER = r"../0Domain_Vocabulary/"

nlp = spacy.load("en_core_web_sm")
ruler = EntityRuler(nlp, overwrite_ents=True)

# Read JSON file
#with open('VOCABULARY.json') as data_file:
with open('VOCABULARY_TYPED.json', encoding="utf8") as data_file:
    patterns = json.load(data_file)

ruler.add_patterns(patterns)
nlp.add_pipe(ruler)

#DICTIONARY_FOLDER = r"C:/Users/20230326/0Domain_Vocabulary/"
### ==================================================================================================
# Tagger
def filter_spans(spans):
    # Filter a sequence of spans so they don't contain overlaps
    get_sort_key = lambda span: (span.end - span.start, span.start)
    sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
    result = []
    seen_tokens = set()
    for span in sorted_spans:
        if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
            result.append(span)
            seen_tokens.update(range(span.start, span.end))
    return result

# for entities
def tag_noun_chunks(doc):
    # entities
    spans = list(doc.ents)
    spans = filter_spans(spans)
    with doc.retokenize() as retokenizer:
        #string_store = doc.vocab.strings
        for span in spans:
            #start = span.start
            #end = span.end            
            #retokenizer.merge(doc[start: end], attrs=intify_attrs({'ent_type': 'NOUN_CHUNK'}, string_store))
            retokenizer.merge(span)
    
    """
    # noun chunks
    spans = list(doc.noun_chunks)
    spans = filter_spans(spans)
    with doc.retokenize() as retokenizer:
        string_store = doc.vocab.strings
        for span in spans:
            start = span.start
            end = span.end            
            retokenizer.merge(doc[start: end], attrs=intify_attrs({'ent_type': 'NOUN_CHUNK'}, string_store))
            #retokenizer.merge(span)
    """
    
# for verbs
def tag_chunks_spans(doc, spans, span_type):
    spans = filter_spans(spans)
    with doc.retokenize() as retokenizer:
        string_store = doc.vocab.strings
        for span in spans:
            start = span.start
            end = span.end
            retokenizer.merge(doc[start: end], attrs=intify_attrs({'ent_type': span_type}, string_store))
            #retokenizer.merge(doc[start: end])

def clean(text):
    text = text.strip('[(),- :\'\"\n]\s*')
    #text = re.sub('([A-Za-z0-9\)]{2,}\.)([A-Z]+[a-z]*)', r"\g<1> \g<2>", text, flags=re.UNICODE)
    text = re.sub('","', ' ', text, flags=re.UNICODE).strip()
    #text = re.sub('\/', ' ', text, flags=re.UNICODE).strip()
    text = re.sub('-', ' ', text, flags=re.UNICODE).strip()
    #text = re.sub('\(', ' ', text, flags=re.UNICODE).strip()
    #text = re.sub('\)', ' ', text, flags=re.UNICODE).strip()
    text = text.replace("\\", ' ')
    text = text.replace("[", ' ')
    text = text.replace("]", ' ')
    text = text.replace("Mt ", 'Mount ')
    text = text.replace("characterize", 'characterise')
    text = text.replace("mineralize", 'mineralise')    
    text = re.sub('\s+', ' ', text, flags=re.UNICODE).strip()
        
    text = ' '.join(text.lower().split())
    
    if (text[len(text)-1] != '.'):
        text += '.'
    
    #print(text, '\n')
    return text

def tagger(text):  
    df_out = pd.DataFrame(columns=['Document#', 'Sentence#', 'Word#', 'Word', 'EntityType', 'EntityIOB', 'Lemma', 'POS', 'POSTag', 'Start', 'End', 'Dependency'])
    text = clean(text)
    
    document = nlp(text)    
    
    # Get the entities in each sentence
    sentences_entities = []
    for s in document.sents:
        ents_s = []
        for e in s.ents:
            if e.label_ in ['ORE_DEPOSIT', 'ROCK', 'MINERAL', 'STRAT', 'LOCATION', 'TIMESCALE']:
                ents_s.append([e.text, e.label_])
        if ents_s:
            sentences_entities.append(ents_s)
    #print('Entities in each sentence:', *sentences_entities, sep='\n')
    
    annotation = []
    #print('Spacy NLP before chunking \n')
    for token in document:
        #print(token.text, token.ent_type_)
        if token.ent_type_ in ['ORE_DEPOSIT', 'ROCK', 'MINERAL', 'STRAT', 'LOCATION', 'TIMESCALE']:
        #print([token.i, token.text, token.ent_type_, token.ent_iob_, token.lemma_, token.pos_, token.tag_, token.idx, token.idx+len(token)-1, token.dep_])
            annotation.append([token.text, token.ent_iob_ + '-' + token.ent_type_]) #, token.pos_, token.tag_]) 
            #print(token.text, token.ent_iob_ + '-' + token.ent_type_)
        else:
            annotation.append([token.text, 'O'])
            #print(token.text, 'O')
    #print('\n\n\n')
    
    # Chunk entities
    tag_noun_chunks(document)    
    
    # Chunk verbs with multiple words: 'were exhibited'
    spans_change_verbs = []
    for i in range(1, len(document)):
        w_left = document[i-1]
        w_right = document[i]
        if w_left.pos_ == 'VERB' and w_right.pos_ == 'VERB' and w_left.ent_type_ != 'NOUN_CHUNK' and w_right.ent_type_ != 'NOUN_CHUNK':
            spans_change_verbs.append(document[w_left.i : w_right.i + 1])
    tag_chunks_spans(document, spans_change_verbs, 'VERB_CHUNK')

    # Chunk verbs
    spans_change_verbs = []
    for i in range(2, len(document)):
        w_left = document[i-2]
        w_middle = document[i-1]
        w_right = document[i]
        if w_left.ent_type_ == 'ADP' and w_middle.ent_type_ == 'VERB' and w_right.ent_type_ == 'ADP' and w_left.ent_type_ != 'NOUN_CHUNK' and w_middle.ent_type_ != 'NOUN_CHUNK' and w_right.ent_type_ != 'NOUN_CHUNK':
            spans_change_verbs.append(document[w_left.i : w_right.i + 1])
    tag_chunks_spans(document, spans_change_verbs, 'VERB_CHUNK')

    # Chunk verbs
    spans_change_verbs = []
    for i in range(2, len(document)):
        w_left = document[i-2]
        w_middle = document[i-1]
        w_right = document[i]
        if w_left.ent_type_ == 'VERB' and w_middle.ent_type_ == 'PART' and w_right.ent_type_ == 'ADP' and w_left.ent_type_ != 'NOUN_CHUNK' and w_middle.ent_type_ != 'NOUN_CHUNK' and w_right.ent_type_ != 'NOUN_CHUNK':
            spans_change_verbs.append(document[w_left.i : w_right.i + 1])
    tag_chunks_spans(document, spans_change_verbs, 'VERB_CHUNK')

    # Chunk: adp + verb; part  + verb
    spans_change_verbs = []
    for i in range(1, len(document)):
        w_left = document[i-1]
        w_right = document[i]
        if w_right.pos_ == 'VERB' and (w_left.pos_ == 'ADP' or w_left.pos_ == 'PART')  and w_left.ent_type_ != 'NOUN_CHUNK' and w_right.ent_type_ != 'NOUN_CHUNK':
            spans_change_verbs.append(document[w_left.i : w_right.i + 1])
    tag_chunks_spans(document, spans_change_verbs, 'VERB_CHUNK')

    # Chunk: verb + adp; verb + part 
    spans_change_verbs = []
    for i in range(1, len(document)):
        w_left = document[i-1]
        w_right = document[i]
        if w_left.pos_ == 'VERB' and (w_right.pos_ == 'ADP' or w_right.pos_ == 'PART')  and w_left.ent_type_ != 'NOUN_CHUNK' and w_right.ent_type_ != 'NOUN_CHUNK':
            spans_change_verbs.append(document[w_left.i : w_right.i + 1])
    tag_chunks_spans(document, spans_change_verbs, 'VERB_CHUNK')

    # Chunk verbs with multiple words: 'were exhibited'
    spans_change_verbs = []
    for i in range(1, len(document)):
        w_left = document[i-1]
        w_right = document[i]
        if w_left.pos_ == 'VERB' and (w_right.pos_ == 'VERB') and w_left.ent_type_ != 'NOUN_CHUNK' and w_right.ent_type_ != 'NOUN_CHUNK':
            spans_change_verbs.append(document[w_left.i : w_right.i + 1])
    tag_chunks_spans(document, spans_change_verbs, 'VERB_CHUNK')

    for i in range(1, len(document)):
        w_single_verb = document[i]
        if w_single_verb.pos_ == 'VERB' and w_single_verb.ent_type_ == '':
            spans_change_verbs.append(document[w_single_verb.i : w_single_verb.i + 1])
    tag_chunks_spans(document, spans_change_verbs, 'VERB_CHUNK')

    # chunk all between LRB- -RRB- (something between brackets)
    start = 0
    end = 0
    spans_between_brackets = []
    for i in range(0, len(document)):
        if ('-LRB-' == document[i].tag_ or r"(" in document[i].text):
            start = document[i].i
            continue
        if ('-RRB-' == document[i].tag_ or r')' in document[i].text):
            end = document[i].i + 1
        if (end > start and not start == 0):
            span = document[start:end]
            try:
                assert (u"(" in span.text and u")" in span.text)
            except:
                pass
                #print(span)
            spans_between_brackets.append(span)
            start = 0
            end = 0
    tag_chunks_spans(document, spans_between_brackets, '')
            
    """
    # chunk entities
    spans_change_ents = []
    for i in range(1, len(document)):
        w_left = document[i-1]
        w_right = document[i]
        if w_left.ent_type_ == 'NOUN_CHUNK' and w_right.ent_type_ == 'NOUN_CHUNK':
            spans_change_ents.append(document[w_left.i : w_right.i + 1])
    tag_chunks_spans(document, spans_change_ents, 'NOUN_CHUNK')

    # chunk entities
    spans_change_ents = []
    for i in range(1, len(document)):
        w_left = document[i-1]
        w_right = document[i]
        if w_left.tag_ == 'DT' and w_right.ent_type_ == 'NOUN_CHUNK':
            spans_change_ents.append(document[w_left.i : w_right.i + 1])
    tag_chunks_spans(document, spans_change_ents, 'NOUN_CHUNK')
    """
    
    ignore_entities = ['drill', 'ltd', 'limited', 'pty', 'company', 'project', 'prospect', 'hole', 'twin', 'impregnate', 'core']
    noun_chunks = []
    doc_id = 0
    count_sentences = 0
    #print("Some Rules to ignore some entities are performed.\n\n")
    for token in document:
        #print(len(document), token.i)
        if (token.text == '.'):
            count_sentences += 1
        elif (token.ent_type_ != '' and token.ent_type_ != 'VERB_CHUNK'):
            noun_chunks.append(token)
        if token.tag_ == 'CD' or token.ent_type_ == 'CARDINAL' or token.ent_type_ == 'QUANTITY':
            #print(token, token.tag_)
            df_out.loc[len(df_out)] = [doc_id, count_sentences, token.i, token.text, '', '', token.lemma_, token.pos_, token.tag_, token.idx, token.idx+len(token)-1, token.dep_]
        elif (len(document) > token.i + 1):
            #print(token.text, token.i, doc[token.i+1].text, token.i + 1)            
            if (bool([w for w in ignore_entities if(w in document[token.i+1].text)])):
                df_out.loc[len(df_out)] = [doc_id, count_sentences, token.i, token.text, '', '', token.lemma_, token.pos_, token.tag_, token.idx, token.idx+len(token)-1, token.dep_]
            else:
                df_out.loc[len(df_out)] = [doc_id, count_sentences, token.i, token.text, token.ent_type_, token.ent_iob_, token.lemma_, token.pos_, token.tag_, token.idx, token.idx+len(token)-1, token.dep_]
        else:
            df_out.loc[len(df_out)] = [doc_id, count_sentences, token.i, token.text, token.ent_type_, token.ent_iob_, token.lemma_, token.pos_, token.tag_, token.idx, token.idx+len(token)-1, token.dep_]
    
    # Get the unique entities for the whole document
    doc_ents = {e.text: e.label_ for e in document.ents if e.label_ in ['ORE_DEPOSIT', 'ROCK', 'MINERAL', 'STRAT', 'LOCATION', 'TIMESCALE']}
    #print("ENTS\n", len(doc_ents), '\n', doc_ents, '\n\n')
    
    return df_out, doc_ents, document, annotation, sentences_entities #, words_ids #noun_chunks

### ==================================================================================================
### triple extractor

def fix_sentences(df_text):
    sentences_all = []
    aSentence = []
    
    for index, row in df_text.iterrows():
        d_id, s_id, word_id, word, ent, ent_iob, lemma, cg_pos, pos, start, end, dep = row.items()
        
        #if 'VERB_CHUNK' == ent[1] or (('ADP' in cg_pos[1] or 'VERB' in cg_pos[1]) and ent[1] == ''):
        if 'VERB_CHUNK' == ent[1] or (('ADP' in cg_pos[1] or 'VERB' in cg_pos[1]) and ent[1] == ''):
            aSentence.append([word[1], word_id[1], 'VERB_CHUNK'])
        #elif 'obj' in dep[1]:
        #    aSentence.append([word[1], word_id[1], 'object'])
        elif ent[1] != '' and ent[1] != 'VERB_CHUNK':
            aSentence.append([word[1], word_id[1], ent[1]]) # 'NOUN_CHUNK'
        elif word[1] == ' ':
                pass
        elif word[1] == '.':
            aSentence.append([word[1], word_id[1], '.'])
            sentences_all.append(aSentence)
            aSentence = []
        else:
            #aSentence.append([word[1], word_id[1], pos[1]])    
            aSentence.append([word[1], word_id[1], ''])
    
    #print('\n', sentences, '\n')    
    return sentences_all # aSentence

def get_predicate(s):
    pred_ids = {}
    for w, index, spo in s:
        #print(s)
        if spo == 'VERB_CHUNK' and w != "'s":
            pred_ids[index] = w
    predicates = {}
    for key, value in pred_ids.items():
        predicates[key] = value
    #print(predicates)
    return predicates
def get_positions(s, start, end):
    adps = {}
    #print(start, end)
    for w, index, spo in s:
        #print(w, index, spo)
        if index >= start and index <= end and not (spo == '' or spo == '.' or spo == 'VERB_CHUNK'):
            #print(index)
            adps[index] = w
    #print('adps = ', adps, '\n')
    return adps

### ==================================================================================================
def extract_sentence_triples(aSentence): # The input is for each sentence    
    if len(aSentence) == 0: 
        return
    sentence_relations = {}
    relations = []
    preds = get_predicate(aSentence) # Get all verbs
    #print("\nPreds = ", preds, '\n')
    
    raw_sent = ' '.join(list(zip(*aSentence))[0]) # get the raw sentence
    #print(raw_sent, '\n')
    if preds:
        if (len(preds) == 1):
            #print("preds = ", preds)
            predicate = list(preds.values())[0]
            #print("predicate = ", predicate)
            #if (len(predicate) < 2):
            #    predicate = 'is'
            #print(s)
            ents = [e[0] for e in aSentence if e[2] != '' and e[2] != '.' and e[2] != 'VERB_CHUNK']
            #print('ents = ', ents)
            for i in range(1, len(ents)):
                relations.append([ents[0], predicate, ents[i]])
                #print('1 - ', relations, '\n')

        pred_ids = list(preds.keys())
        pred_ids.append(aSentence[0][1])
        pred_ids.append(aSentence[len(aSentence)-1][1])
        pred_ids.sort()
                
        for i in range(1, len(pred_ids)-1):                
            predicate = preds[pred_ids[i]]
            #print('---- predicate = ', predicate)
            
            adps_subjs = get_positions(aSentence, pred_ids[i-1], pred_ids[i])
            #print('- subjects = ', adps_subjs)
            adps_objs = get_positions(aSentence, pred_ids[i], pred_ids[i+1])
            #print('- objects = ', adps_objs)
            
            for k_s, subj in adps_subjs.items():
                for k_o, obj in adps_objs.items():
                    obj_prev_id = int(k_o) - 1
                    if obj_prev_id in adps_objs: # at, in, of
                        #relations.append([subj, predicate + ' ' + adps_objs[obj_prev_id], obj, raw_sent])
                        relations.append([subj, predicate, obj])
                        #print('2 - ', relations, '\n')
                    else:
                        relations.append([subj, predicate, obj])                        
                        #print('3 - ', relations, '\n')
    sentence_relations[raw_sent] = relations
    
    return sentence_relations

def extract_triples_by_sentences(sentences):
    remove_words = ['a', 'an', 'the', 'its', 'their', 'his', 'her', 'our', 'who', 'that', 'this', 'these', 'those']
    triples_doc = []
    
    #print('Num sentences = ', len(sentences))    
    whole_text = ' '.join(sentences)
    
    df_tagged, entities_doc, doc, annotation, entities_by_sentences = tagger(whole_text)
    #print('tagger function:\n', entities_doc)
    
    # Get sentences
    sentences_all = fix_sentences(df_tagged) 
    
    for s in sentences_all:
        #print('\nDATAFRAME:\n', s, '\n')
        # Extract relations(s)
        sentence_triples = extract_sentence_triples(s)
        
        for raw_sentence, triples in sentence_triples.items():  
            #print('\nSENTENCE:\n', raw_sentence, '\n')
            filtered_triples = []
            for s, p, o in triples:
                if s == o:
                    continue            
                subj = s.strip('[,- :\'\"\n]*')
                pred = p.strip('[- :\'\"\n]*.')
                obj = o.strip('[,- :\'\"\n]*')
                
                subj = ' '.join(word for word in subj.lower().split() if not word in remove_words)
                obj = ' '.join(word for word in obj.lower().split()  if not word in remove_words)
                subj = re.sub("\s\s+", " ", subj)
                obj = re.sub("\s\s+", " ", obj)
                
                if subj and pred and obj:
                    filtered_triples.append([subj, pred, obj])
            #print('TRIPLES:\n', filtered_triples, '\n')
            triples_doc = triples_doc + filtered_triples

    return triples_doc, entities_doc, entities_by_sentences

# Synonyms and Abbreviations are siolved using a list in synonyms.csv file
def resolve_synonyms(g):
    synonyms = {}
    with open(DICTIONARY_FOLDER + "synonyms.csv", 'r', encoding='latin1') as f:        
        reader = csv.reader(f, delimiter=',')
        for row in reader:            
            synonyms[row[0].lower()] = row[1].lower() # abbreviation and exact words
    #print('synonyms # : ', synonyms)
    
    remove_nodes = []
    add_edges = []
    add_nodes = []
    #print('Resolve synonym before:\n', nx.info(g))
    for s, t, data in g.edges(data=True):
        #if s not in remove_nodes:
            if s in synonyms.keys():
                if synonyms[s] not in g.nodes():
                    add_nodes.append([synonyms[s], g.nodes[s]])
                add_edges.append([synonyms[s], t, data['label']])
                #print('add_edge: ', [synonyms[s], t, data['label']])
                if s not in remove_nodes:
                    remove_nodes.append(s)
        #if t not in remove_nodes:
            if t in synonyms.keys():
                if synonyms[t] not in g.nodes():
                    add_nodes.append([synonyms[t], g.nodes[t]])
                add_edges.append([s, synonyms[t], data['label']])
                #print('add_edge: ', [s, synonyms[t], data['label']])
                if t not in remove_nodes:
                    remove_nodes.append(t)
    
    # deal with unconnected nodes
    for n, data in g.nodes(data=True):
        if n not in remove_nodes:
            if n in synonyms.keys():
                if synonyms[n] not in g.nodes():
                    add_nodes.append([synonyms[n], g.nodes[n]])
                remove_nodes.append(n)
                
    for n, data in add_nodes:
        #print('add node: ', n, data)
        g.add_node(n, group=data['group'])
    
    for s, t, label in add_edges:
        #print('add egde: ', s, t, d)
        #g.add_edge(s, t, label=d)
        if g.has_edge(s, t) and label:
            current_labels = g[s][t]['label'].split(',')
                #print(current_labels)
            if label not in current_labels:
                g.add_edge(s, t, label=g[s][t]['label'] + ',' + label)
        else:
            #print('- add egde: ', [s], [t], [label])        
            g.add_edge(s, t, label=label)
    
    if len(remove_nodes) > 0:
        #print('Synonyms are found. Remove nodes : ', remove_nodes)
        for n in remove_nodes: # remove the merged nodes
            if n in g.nodes():
                g.remove_node(n)
    
    #print('Resolve synonym after:\n', nx.info(g))
    return g


# Plural to singular
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
        if g.has_edge(s, t) and label:
            current_labels = g[s][t]['label'].split(',')
                #print(current_labels)
            if label not in current_labels:
                g.add_edge(s, t, label=g[s][t]['label'] + ',' + label)
        else:
            #print('- add egde: ', [s], [t], [label])        
            g.add_edge(s, t, label=label)
    
    for n in remove_nodes: # remove the merged nodes
        if n in g.nodes():
            #print('--- remove node: ', n)
            g.remove_node(n)
    
    #print('Resolved plurals after:\n', nx.info(g))
    return g

def get_graph(triples):
    G = nx.DiGraph()
    for s, p, o in triples:
        G.add_edge(s, o, label=p)
    return G

def draw_graph_centrality(G):    
    degree_dict = dict(G.degree(G.nodes()))
    nx.set_node_attributes(G, degree_dict, 'degree')
    
    plt.figure(figsize=(15,8))
    pos = nx.spring_layout(G)
    #print("Nodes\n", G.nodes(True))
    #print("Edges\n", G.edges())
    nx.draw_networkx_nodes(G, pos, 
            nodelist=degree_dict.keys(),
            with_labels=False,
            edge_color='black',
            width=1,
            linewidths=1,
            node_size = [v * 100 for v in degree_dict.values()],
            node_color='blue',
            alpha=0.3)
    edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
    #print(edge_labels)
    nx.draw_networkx_edge_labels(G, pos,
                           font_size=9,
                           edge_labels=edge_labels,
                           font_color='red')
    nx.draw(G, pos, with_labels=True, node_size=1, node_color='blue')

def draw_graph(G):
    plt.figure(figsize=(12,10))
    pos = nx.spring_layout(G)
    #print("Nodes\n", G.nodes(True))
    #print("Edges\n", G.edges())
    
    edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
    #print(edge_labels)
    nx.draw_networkx_edge_labels(G, pos,
                           font_size=10,
                           edge_labels=edge_labels,
                           font_color='blue')
    nx.draw(G, pos, with_labels=True, node_size=4, node_color='blue')

def build_graphs(triples, entities):        
    G = nx.DiGraph()
    #print(f'\nBuilding a graph for {len(triples)} number of triple.\n') 
    
    for s, p, o in triples:
        try:            
            #print('---- triple: ', s, p, o)
            # create a node
            if (s not in G.nodes()):
                #print('- subject: ', s, entities[s])
                G.add_node(s, group=entities[s])
            if (o not in G.nodes()):
                G.add_node(o, group=entities[o])
                #print('- object: ', o, entities[o])
            if G.has_edge(s, o):
                current_labels = G[s][o]['label'].split(',')
                #print('labels: ', current_labels)
                if p.strip() not in current_labels:
                    G.add_edge(s, o, label=G[s][o]['label'] + ',' + p)
            else:
                G.add_edge(s, o, label=p)
        except:
            pass
    #print('NODES\n', G.nodes(data=True))
    #print('EDGES\n', G.edges(data=True))
    return G, entities

# Create a graph file from a given graph
def create_graph_file(g, filename):
    # Add degree
    degree_dict = dict(g.degree(g.nodes()))
    nx.set_node_attributes(g, degree_dict, 'degree')

    # Save graph files
    #nodes = [{'id': n, 'group': g.nodes[n]['group'], 'degree': g.nodes[n]['degree']} for n in g.nodes()]
    nodes = [{'id': n, 'group': g.nodes[n]['group'], 'degree': str(g.nodes[n]['degree'])} for n in g.nodes()]
    #links = [{'source': u, 'target': v, 'label': data['label'], 'data': data} for u, v, d in graph.edges(data=True)]
    links = [{'source': u, 'target': v, 'label': d['label']} for u, v, d in g.edges(data=True)]
    with open(filename, 'w') as f:
        json.dump({'nodes': nodes, 'links': links}, f, indent=4,)


def join_graphs(big_g, small_g):
    #print('\nJoin graph with Strats:')
    G = nx.DiGraph()
    
    for n,d in big_g.nodes(data=True):
        if not G.has_node(n):  
            G.add_node(n, group = d['group'])
    for n,d in small_g.nodes(data=True):
        if not G.has_node(n):  
            G.add_node(n, group = d['group'])

    all_edges = {} # key=[s,t]; value=list of relations
    for s, t, d in big_g.edges(data=True):
        k = s+';'+t
        all_edges[k] = d['label'].split(',')
        #print('*** add edge: ', k, all_edges[k])
    
    for s, t, d in small_g.edges(data=True):
        k = s+';'+t
        #print('--- ', k)
        if k in all_edges.keys():
            #print('\nBig_G: ', all_edges[k], '\nSmall_G: ', d['label'].split(','))
            all_edges[k] = all_edges[k] + d['label'].split(',')
            #print('\nJoined: ', all_edges[k])        
        else:
            all_edges[s+';'+t] = d['label'].split(',')
        
    for key,value in all_edges.items():
        #print([key], [value])
        s, t = key.split(';')
        labels = list(set(value))
        labels = [l.strip() for l in labels]
        labels = [l for l in labels if l]
        G.add_edge(s, t, label=','.join(labels))
            
    return G

def text2graph(text):
    
    sentences = text.split('","')
    #$print('Num sentences = ', len(sentences))
    
    doc_triples, doc_entities, entities_by_sentences= extract_triples_by_sentences(sentences)
    #print('# of triples', len(doc_triples))
    #print('Num sentences with triples = ', len(Sent_Triples))

    #print("\n\nEntities:", len(entities), "\n")
    #for e in entities:
    #    print(e, e.label_, e.start, e.end)
    
    # Create entities from the entities of each sentence & connect all entities in a single sentence (add to triple)  
    #print('******************************')    
    for s_ents in entities_by_sentences:
        for i in range(0, len(s_ents)-1):
            #print(s_ents[i][0], '', s_ents[i+1][0])
            doc_triples.append([s_ents[i][0], '', s_ents[i+1][0]])
    #print('# of triples', len(doc_triples))
        
    g, ents = build_graphs(doc_triples, doc_entities)
    
    F = resolve_synonyms(g)    
    
    # Add Stratigraphic Hierarchical information
    stratList = set([e for e,d in F.nodes(data=True) if d['group'] == 'STRAT'])
    #print('\nSTRATS\n', stratList)
    StratG = get_stratigraphic_hierarchy(stratList)
    #print("\n------ STRATIGRAPHY graph is retrieved. \n") #, nx.info(StratG))
   
    # Merge main graph G with the straigraphic graph StratG
    M = join_graphs(F, StratG)    
    M.remove_edges_from(nx.selfloop_edges(M)) 
    
    N = resolve_plurals(M)
    N.remove_edges_from(nx.selfloop_edges(N))
    
    # Clean labels for edges
    for s, t, data in N.edges(data=True):
        N[s][t]['label'] = N[s][t]['label'].strip(',')
        
    return N


if __name__ == "__main__":
    
    text = """The Coolgardie district is located on the western side of the Archean Menzies-Norseman Greenstone Belt, Eastern Goldfields Province in Archaean.","Swager et al.","(1990) subdivide the Archean Menzies-Norseman Greenstone Belt into the Kalgoorlie Terrane, Menzies Terrane and the Norseman Terrane.","The Kalgoorlie Terrane is further subdivided into six tectono-stratigraphic domains: Bullabulling, Coolgardie, Ora Banda, Kambalda, Boorara and the Parker domains. Archaean is a tinme scale."""
    
    g = text2graph(text)
    #print(len(annotation), annotation)
    #g = text2graph(text)
    draw_graph_centrality(g)    
    
    print("\n\nNODES:")
    for n in g.nodes(data=True):
        print(n)
    print("\n\nEDGES:")
    for s,t,a in g.edges(data=True):
        print([s], [t], ' --- ', a['label'])
     
    print("\nTyped grap - ", nx.info(g))   
    print("\nFinished the process.")