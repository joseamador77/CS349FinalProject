from LoadData import gather_clean_data, train_test_split
from sklearn import neighbors, datasets
from sklearn.metrics import accuracy_score
import csv
import time




startTime = time.time()
train_features, train_targets, test_features, test_targets = gather_clean_data('data/cleanedData.csv',0.9)
#train_features, train_targets, test_features, test_targets = gather_clean_data('data/smallerData.csv',0.9)
n_neighbors = 20
print('got data')

clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
clf.fit(train_features,train_targets)
print('Finished Fitting')
predictedVals = clf.predict(test_features)
print('Finished predicting')
wow = accuracy_score(test_targets,predictedVals)

print(wow)
endTime = time.time()


with open('data/predicted_targets.csv','w',newline = '') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in predictedVals:
        writer.writerow([i])

with open('data/test_targets.csv','w',newline = '') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in test_targets:
        writer.writerow([i])



print('DONE######################################################')
totalTime = endTime-startTime
print('Minutes =',totalTime/60)









