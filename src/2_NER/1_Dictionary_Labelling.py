# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 09:18:37 2020

@author: majiga
"""
import glob, os, json
import pandas as pd
#import numpy as np
import csv
import time
import networkx as nx
import matplotlib.pyplot as plt
   
from graph_from_file import tagger

#foldername = r"C:/Users/20230326/1-2 OzROCK/Files/"
foldername = r"Files/OzROCK dataset/"
    #data = pd.read_csv(filename, header = None, delimiter=" ", na_values=['\n'], quoting=csv.QUOTE_NONE, encoding='latin1', skip_blank_lines=True)
    #words = data.iloc[:,0]
    #text = ' '.join([str(w) for w in words])


### 1. TRAIN, VALIDATION and TEST SETS #########

# Create a sentences
data = pd.read_csv(foldername + "Dictionary_labelled_dataset_for_NER_model_preparation.txt", header = None, delimiter=" ", 
                   na_values=['\n'], quoting=csv.QUOTE_NONE, encoding='latin1', skip_blank_lines=True)
words = data.iloc[:,0]
text_big_data = ' '.join([str(w) for w in words])
sentences_big_data = text_big_data.split(' . ')
sentences_big_data = [s.strip()+' .' for s in sentences_big_data]

# Save the sentences to a text file
with open(foldername + "BIG_set_sentences.txt", 'w') as f:
    for s in sentences_big_data:
        f.write(s + '\n')
print('Num sentences in BIG dataset = ', len(sentences_big_data))





def label_file(file_in, file_out):
    
    start_time = time.time()
    fh = open(file_out,"w")

    with open(file_in, 'r') as f:
        sentences = f.readlines()
    
    sentences = [x.strip() for x in sentences]
    print('Num sentences = ', len(sentences))
    count = 0

    for s in sentences:    
        try:
            if len(s.split()) > 100:
                print('--- too long: ', count, '   ', s)
                continue            
            if len(s.split()) < 5:
                print('--- too short: ', count, '   ', s)
                continue

            df, ents, doc, annotation = tagger(s)
            
            annotated = False            
            word_tag_list = []
            for word, tag in annotation:
                if tag != 'O':
                    #print(word, tag)
                    annotated = True
                word_tag_list.append([word, tag])
            
            if annotated:
                for word,tag in word_tag_list:
                    fh.write(word + ' ' + tag + '\n')
                    if word == '.':
                        fh.write('\n')
            else:            
                print('--- not annotated: ', count, '   ', s)
                continue
            
            count += 1            
            if count % 1000 == 0:
                print(count)   
        
            #if count == 20:
            #    break
        except:
            print('--- exception: ', count, '   ', s)
            
    fh.close()
    print("Number of annotated sentences: ", count)
    print("Duration: ", (time.time() - start_time)/60, ' mins.')    


### TRAIN, VALIDATION and TEST SETS #########
    
# Label auto-labelling train, validation and test sets
#label_file(foldername + "autolabelled_old.txt", foldername + "autolabelled_dictionary_labelled.txt")
label_file(foldername + "BIG_set_sentences.txt", foldername + "BIG_set_byDictionary.txt")






### 2. EVALUATION SET #########

# 2.1. Prepare the evaluation sets as a list of sentences in a text file
data = pd.read_csv(foldername + "EvaluationSet_annotated_by_doman_experts.txt", 
                   usecols = ['Word', 'TrueLabel'],
                   delimiter=" ", 
                   na_values=['\n'], quoting=csv.QUOTE_NONE, encoding='latin1', skip_blank_lines=True)
#words = data.iloc[:,0]
words = data['Word'].tolist()
text_evaluation = ' '.join([str(w) for w in words])
sentences_evaluation = text_evaluation.split(' . ')
sentences_evaluation = [s.strip()+' .' for s in sentences_evaluation]

# Save the sentences to a text file
with open(foldername + "EvaluationSet_sentences.txt", 'w') as f:
    for s in sentences_evaluation:
        f.write(s + '\n')
print('Num sentences in Evaluation dataset = ', len(sentences_evaluation))


# 2.2. Dictionary-labelling EVALUATION dataset

def dictionary_labelling_evaluationSet(file_in, file_out):
    
    start_time = time.time()
    print('Started: ', start_time)
    fh = open(file_out,"w")
    fh.write('Word DictionaryLabel\n') # Add header

    with open(file_in, 'r') as f:
        sentences = f.readlines()
    
    sentences = [x.strip() for x in sentences]
    print('Num sentences = ', len(sentences))
    count = 0

    for s in sentences:    
        try:
            if len(s.split()) > 100:
                print('--- too long: ', count, '   ', s)
                #continue            
            if len(s.split()) < 5:
                print('--- too short: ', count, '   ', s)
                #continue

            df, ents, doc, annotation = tagger(s)
            annotated = False            
            for word, tag in annotation:
                if tag != 'O':
                    #print(word, tag)
                    annotated = True            
                fh.write(word + ' ' + tag + '\n')
                if word == '.':
                    fh.write('\n')
        
            if annotated == False:
                print('--- not annotated: ', count, '   ', s)
            count += 1           
            if count%500 == 0:
                print(count)        
            #if count == 20:
            #    break
        except:
            print('--- exception: ', count, '   ', s)
            
    fh.close()
    print("Number of annotated sentences: ", count)
    print("Duration: ", (time.time() - start_time)/60, ' mins.')    

# Dictionary-Label the evaluation file
#sentences_evaluation = [str(s) + "." for s in sentences_evaluation]
dictionary_labelling_evaluationSet(foldername + "EvaluationSet_sentences.txt", foldername + "EvaluationSet_byDictionary.txt")




# Check EVALUATION SET F1 score accuracy between the ground truth and the dictionary-labels
from seqeval.metrics import classification_report  #, precision_score, recall_score, f1_score

# Groubnd truth labels
with open(foldername + "EvaluationSet_annotated_by_doman_experts.txt", 'r') as f:
    print(f.readline())
    sentences_true = f.readlines()
    print('Num lines in ground truth = ', len(sentences_true))
# Dictionary labels
with open(foldername + "EvaluationSet_byDictionary.txt", 'r') as f:
    print(f.readline())
    sentences_dict = f.readlines()
    print('Num lines in dictionary-labelling', len(sentences_dict))
    
words = []
labels_true = []
labels_predicted = []
c = 0
for strue, sdict in zip(sentences_true, sentences_dict):
    try:
        c += 1
        #print(strue, '\n', sdict)
        if strue.strip() == '':
            continue
        word, true_label = strue.split()
        words.append(word.strip())
        labels_true.append(true_label.strip())
        
        word_dict, label_dict = sdict.split()
        if word.strip().lower() == word_dict.strip().lower():
            labels_predicted.append(label_dict.strip())
        else:
            print('Error: ', c, word)
            break
        if true_label.strip().lower() != label_dict.strip().lower():
            print('Diff: ', c, word, true_label, label_dict)
    except:
        print('Exception: ', c, strue, sdict)



print(len(labels_true))
#print(y_te[0])

print(len(labels_predicted))
#print(y_pred[0])

#print(classification_report(labels_true, labels_predicted))
print(classification_report(labels_true, labels_predicted, digits=2))




"""
remove = ["Sentence not annotated: 7 table 3 1 : geological and topographic wireframes for burbanks",
"Sentence not annotated: sulphur is variable across the project area",
"Sentence not annotated: sons of gwalia ltd explored extensively to the west of e40 184 for au between 1992 and 2002"]

# Create a file with raw sentences.
f_sentences = open(foldername + "autolabelled_smallsize_sentences.txt","w")
for s in sentences:
    a = 'Sentence not annotated: ' + s
    #print(a)
    
    if (a in remove):
        continue
    
    f_sentences.write(s + '.\n')
    count += 1
fh.close()
f_sentences.close()
"""



"""
with open('wamex_files.csv', mode='w', newline='', encoding='utf-8') as file:
    csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #csv_writer.writerow(['head', 'relation', 'tail', head_group, tail_group])
    csv_writer.writerows(files_ids)
print("wamex_files.csv file is created.")
"""


