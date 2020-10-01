# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:08:19 2020

@author: 20230326
"""
import csv
import pandas as pd
import numpy as np

foldername = r"Files/OzROCK dataset/"

filename_dictionary = foldername + "BIG_Set_byDictionary.txt"
filename_model = foldername + "BIG_Set_byModel.txt"
filename_merged = foldername + "BIG_Set_merged.txt"


# TVT set - Dictionary labelling
df_dictionary = pd.read_csv(filename_dictionary, quoting=csv.QUOTE_NONE, 
                 header=None, encoding='Latin1', skip_blank_lines=True, sep=' ')
#df = df.replace(np.nan, '', regex=True)
print(len(df_dictionary))
print("AUTO-LABELLED DATASET - DICTIONARY LABELLING: \n", df_dictionary[1].value_counts())

# TVT set - MODEL labelling
df_model = pd.read_csv(filename_model, delimiter=" ", usecols=['Word', 'DictionaryTag', 'ModelTag'],
                       encoding='utf-8', skip_blank_lines=False,
                       quoting=3) # header=None, 
df_model = df_model.replace(np.nan, '', regex=True)
#print(df_model.head(3))
print(len(df_model))
print("\nAUTO-LABELLED DATASET - DICTIONARY LABELLING: \n", df_model['DictionaryTag'].value_counts())
print("\nAUTO-LABELLED DATASET - MODEL LABELLING: \n", df_model['ModelTag'].value_counts())



"""
Merge dictionary and model labels
"""
c, index = 0, 0
labels_dict = df_model['DictionaryTag'].tolist() 
words_model = df_model['Word'].tolist()
labels_model = df_model['ModelTag'].tolist()
labels_merged = []

with open(filename_merged, 'w') as f:
    f.write('Word DictionaryTag ModelTag MergedTag\n')
    for word,tag_dict, tag_model in zip(words_model, labels_dict, labels_model):
        index += 1
        if word == '':                
            f.write('\n')
            labels_merged.append('')
            continue
        td = str(tag_dict)
        tm = str(tag_model)
        #print('\nword, td, tm', word1, td, tm)
        if td == tm:                
            f.write(word + ' ' + td + ' ' + tm + ' ' + td + '\n')
            labels_merged.append(td)
            continue
        if td == 'O' and tm is not 'O':
            #df.loc[index, 'Tag'] = row[2]
            f.write(word + ' ' + td + ' ' + tm + ' ' + tm + '\n')
            labels_merged.append(tm)
            continue
        if td is not 'O' and tm == 'O':
            #df.loc[index, 'Tag'] = row[1]      
            f.write(word + ' ' + td + ' ' + tm + ' ' + td + '\n')
            labels_merged.append(td)
            continue
        if td != tm:            
            if 'I-' in td and (tm == 'O'):
                f.write(word + ' ' + td + ' ' + tm + ' ' + td + '\n')
                tag = td
            elif ('0' == td) and 'I-' in tm:
                f.write(word + ' ' + td + ' ' + tm + ' ' + tm + '\n')
                tag = tm
            else:
                f.write(word + ' ' + td + ' ' + tm + ' ' + td + '\n')
                tag = td
            labels_merged.append(tag)
            if 'B-' in tag:
                c += 1
                print(c, index, word, tag_dict, tag_model)
print("Conflicts # = ", c)


# Merged statistics
df = pd.read_csv(filename_merged, quoting=csv.QUOTE_NONE,
                 usecols=['Word', 'DictionaryTag', 'ModelTag', 'MergedTag'],
                 encoding='Latin1', skip_blank_lines=True, sep=' ')
df = df.replace(np.nan, '', regex=True)
print(df.head(3))
print(len(df))

print("Dictionary: \n", df['DictionaryTag'].value_counts())
print("\n\nPrediction: \n", df['ModelTag'].value_counts())
print("\n\nMerged: \n", df['MergedTag'].value_counts())


# Create a AutoLabelledSet for OzROCK on github.
df = pd.read_csv(filename_merged, quoting=csv.QUOTE_NONE,
                 usecols=['Word', 'DictionaryTag', 'ModelTag', 'MergedTag'],
                 encoding='Latin1', skip_blank_lines=False, sep=' ')
df = df.replace(np.nan, '', regex=True)
print(df.head(3))
print(len(df))

df.loc[df['Word'] == ''].count().iloc[0]


df_combined = pd.DataFrame(df, columns = ['Word', 'MergedTag'])
df_combined.to_csv(foldername + 'AutoLabelledSet.txt', encoding='Latin1', sep=' ', index=False)


# Create a AutoLabelledSet for OzROCK on github.
df = pd.read_csv(foldername + 'AutoLabelledSet.txt', quoting=csv.QUOTE_NONE,
                 usecols=['Word', 'MergedTag'],
                 encoding='Latin1', skip_blank_lines=False, sep=' ')
df = df.replace(np.nan, '', regex=True)
print(df.head(3))
print(len(df))

df.loc[df['Word'] == ''].count().iloc[0]

