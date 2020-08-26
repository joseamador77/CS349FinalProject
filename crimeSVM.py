from LoadData import gather_clean_data, train_test_split
from sklearn import neighbors, datasets
from sklearn import svm
from sklearn.metrics import accuracy_score
from LoadData import train_test_split
import csv
import time




startTime = time.time()
train_features, train_targets, test_featuresFAKE, test_targetsFAKE = gather_clean_data('data/cleanedData.csv',0.5)
test_features, test_targets, test_featuresFAKE2, test_targetsFAKE2 = train_test_split(test_featuresFAKE,test_targetsFAKE, 0.15)
print('got data')

clf = svm.SVC()
clf.fit(train_features,train_targets)
print('Finished Fitting')
predictedVals = clf.predict(test_features)


print('Finished predicting')
wow = accuracy_score(test_targets,predictedVals)
endTime = time.time()
print(wow)


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








