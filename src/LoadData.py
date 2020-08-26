import csv
import numpy as np
import random





def gather_clean_data(path,fraction):
     with open(path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=' ')
        attribute_names = next(csv_reader)

        K = 6
        listVersion = list(csv_reader)
        N = len(listVersion)

        features = np.zeros((N,K))
        targets = np.zeros(N)
        
        for lineNum, feats in enumerate(listVersion):
            targets[lineNum] = feats[-1]
            features[lineNum] = feats[1:len(feats)-1]#dont grab target and ID
        print('Loaded, now splitting')
        return train_test_split(features,targets,fraction)
    

def train_test_split(features, targets, fraction):
    if (fraction > 1.0):
        raise ValueError('N cannot be bigger than number of examples!')
    elif(fraction == 1):
        return features, targets, features, targets
    else:
        N = int(features.shape[0] * fraction)
        M = features.shape[0] - N
        K = features.shape[1]
        train_features = np.zeros((N,K))
        train_targets = np.zeros(N)
        test_features = np.zeros((M,K))
        test_targets = np.zeros(M)
        
        possIndicies = list(range(features.shape[0]))
        random.shuffle(possIndicies)
        
        print('Gathering Train Feats')
        for i in range(N):
            #randPos = np.random.randint(0,features.shape[0])#random position in features/targets
            randPos = possIndicies[i]
            train_features[i] = features[randPos]
            train_targets[i] = targets[randPos]
            
        print('Gathering Test feats')
        
        for j in range(M):
            randPos = possIndicies[i+j+1]
            test_features[j] = features[randPos]
            test_targets[j] = targets[randPos]


    return train_features, train_targets, test_features, test_targets