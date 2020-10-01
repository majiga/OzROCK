#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 22:30:23 2020

@author: majiga

Merge tags: Dictionary-labelled Tags + Model-predicted Tags
"""

import pandas as pd
import numpy as np
import csv

##### EVALUATION set ####
eval_experts = r"Files/OzROCK dataset/EvaluationSet_annotated_by_doman_experts.txt"
eval_dictionary = r"Files/OzROCK dataset/EvaluationSet_byDictionary.txt"
eval_model = r"Files/OzROCK dataset/EvaluationSet_trueTag_modelTag.txt"
eval_merged = r"Files/OzROCK dataset/EvaluationSet_trueTag_modelTag_mergedTag.txt"

"""
#merge_labels(eval_dictionary, eval_model, eval_merged)

df_dictionary = pd.read_csv(eval_dictionary, delimiter=" ", usecols=['Word', 'DictionaryLabel'],
                            encoding='utf-8', skip_blank_lines=False) # header=None,
df_dictionary = df_dictionary.replace(np.nan, '', regex=True)
#print(df_dictionary.head(3))
print(len(df_dictionary))

df_model = pd.read_csv(eval_model, delimiter=" ", usecols=['Word', 'GroundTruth', 'Predicted'],
                       encoding='utf-8', skip_blank_lines=False) # header=None, 
df_model = df_model.replace(np.nan, '', regex=True)
#print(df_model.head(3))
print(len(df_model))

#df["Tag"] = "O"
c = 0
words_dict = df_dictionary['Word'].tolist()
labels_dict = df_dictionary['DictionaryLabel'].tolist() 
words_model = df_model['Word'].tolist()
labels_model = df_model['Predicted'].tolist()
labels_groundtruth = df_model['GroundTruth'].tolist()
labels_merged = []

with open(eval_merged, 'w') as f:
    f.write('Word DictionaryTag ModelTag MergedTag')
    for word1,tag_dict, word2,tag_model in zip(words_dict, labels_dict, words_model, labels_model):
        if word1 != word2:
            print('Words are different: ', word1, word2, tag_dict, tag_model, '\n')
            #break           
        if word1 == '':                
            f.write('\n')
            labels_merged.append('')
            continue
        td = str(tag_dict)
        tm = str(tag_model)
        #print('\nword, td, tm', word1, td, tm)
        if td == tm:                
            f.write(word1 + ' ' + td + ' ' + tm + ' ' + td + '\n')
            labels_merged.append(td)
            continue
        if td == 'O' and tm is not 'O':
            #df.loc[index, 'Tag'] = row[2]
            f.write(word1 + ' ' + td + ' ' + tm + ' ' + tm + '\n')
            labels_merged.append(tm)
            continue
        if td is not 'O' and tm == 'O':
            #df.loc[index, 'Tag'] = row[1]      
            f.write(word1 + ' ' + td + ' ' + tm + ' ' + td + '\n')
            labels_merged.append(td)
            continue
        if td != tm:            
            c += 1
            if 'I-' in td and tm == 'O':
                f.write(word1 + ' ' + td + ' ' + tm + ' ' + td + '\n')
                labels_merged.append(td)
            elif '0' == td and 'I-' in tm:
                f.write(word1 + ' ' + td + ' ' + tm + ' ' + tm + '\n')
                labels_merged.append(tm)
            else:
                f.write(word1 + ' ' + td + ' ' + tm + ' ' + td + '\n')
                labels_merged.append(td)
                
            print("Conflict here ========== ", word1, word2, tag_dict, tag_model)

print("Conflicts # = ", c)
"""
      
      
from seqeval.metrics import classification_report

df_expert = pd.read_csv(eval_experts, 
                   usecols = ['Word', 'TrueLabel'],
                   delimiter=" ", 
                   na_values=['\n'], quoting=csv.QUOTE_NONE, encoding='latin1', skip_blank_lines=True)
df_expert = df_expert.replace(np.nan, '', regex=True)
#print(df_dictionary.head(3))
print(len(df_expert))
words_expert = df_expert['Word'].tolist()
labels_expert = df_expert['TrueLabel'].tolist() 


df_merged = pd.read_csv(eval_merged,
                        usecols=['Word', 'DictionaryTag', 'ModelTag', 'MergedTag'], # Word DictionaryTag ModelTag MergedTag
                        delimiter=" ",                        
                        encoding='utf-8', skip_blank_lines=True)
df_merged = df_merged.replace(np.nan, '', regex=True)
#print(df_model.head(3))
print(len(df_merged))
labels_merged = df_merged['MergedTag'].tolist() 

#labels_expert = [x for x in labels_expert if x]
#labels_merged = [x for x in labels_merged if x]
#sklearn.metrics.classification_report(y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division='warn')
print(classification_report(labels_expert, labels_merged, digits=4)) #, average='micro')) #, digits=2))


labels_model = df_merged['ModelTag'].tolist() 
print(classification_report(labels_expert, labels_model, digits=4))

