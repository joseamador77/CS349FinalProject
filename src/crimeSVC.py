from LoadData import gather_clean_data, train_test_split
from sklearn import neighbors, datasets
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from LoadData import train_test_split
import csv
import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler




startTime = time.time()
train_features, train_targets, test_features, test_targets = gather_clean_data('data/cleanedData.csv',0.7)
print('got data')

clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5))
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








