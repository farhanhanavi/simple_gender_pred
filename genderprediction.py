import numpy as np

len_vocab = 28

def set_flag(i):
    tmp = np.zeros(len_vocab)
    tmp[i] = 1
    return list(tmp)

def prepare_X(X, char_to_index_pickle):
    new_list = []
    trunc_train_name = [str(i)[0:40] for i in X]

    for i in trunc_train_name:
        tmp = [set_flag(char_to_index_pickle[j]) for j in str(i)]
        for k in range(0,40 - len(str(i))):
            tmp.append(set_flag(char_to_index_pickle["END"]))
        new_list.append(tmp)

    return new_list
