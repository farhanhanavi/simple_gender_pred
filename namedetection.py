import pandas as pd
import numpy as np
import pickle

total_feature              = 14631
total_class_feature_nonama = 11243
total_class_feature_nama   = 13554


''' 

Predict probability of being each class

'''


def predict_prob(sentence, freq_list, total_class_feature, total_feature):
    new_word_list = sentence.split()
    
    #Probs of noname
    prob_s_with_ls = []
    for word in new_word_list:
        if word in freq_list.keys():
            count = freq_list[word]
        else:
            count = 0
        prob_s_with_ls.append((count + 1)/(total_class_feature + total_feature))
    return np.prod(prob_s_with_ls)


''' 

Compare and get final result

'''

def detect_name(sentence, noname_frequency_list, name_frequency_list):
    noname_pred = predict_prob(sentence, noname_frequency_list, total_class_feature_nonama, total_feature)
    name_pred   = predict_prob(sentence, name_frequency_list, total_class_feature_nama, total_feature)
    
    if np.argmax([noname_pred, name_pred]) == 1:
        return 'Name'
    else:
        return 'No Name'