import numpy as np
import csv


def write_data(path,dataLines,attribute_names):
    with open(path,'w',newline = '') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(attribute_names)
        for lineNum, line in enumerate(dataLines):
            writer.writerow(line)

def combineIUCR(code):
    

    return 9



def clean_data(path):
    #K features, N datapoints
    with open(path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        attribute_names = next(csv_reader)

        K = 8 #7 attributes and 1 target
        listVersion = list(csv_reader)
        N = len(listVersion)

        features = np.zeros((N,K))
        #targets = np.zeros(N)
        
        
        for lineNum, feats in enumerate(listVersion):
            if lineNum % 100000 == 0:
                print(lineNum)
            
            if feats[19] != '' and feats[13] != '' and feats[11] != '':
                try:
                    features[lineNum][7] = feats[14]
                except ValueError:
                    feats[14] = feats[14][0:len(feats[14])-1]#remove letters from FBI Code
                    features[lineNum][7] = feats[14]
                features[lineNum][0] = feats[0] 
                features[lineNum][3] = feats[11]
                features[lineNum][4] = feats[13]
                features[lineNum][5] = feats[19]
                features[lineNum][6] = feats[20]

                date = feats[2]
                hour = int(date[11:13])
                month = int(date[0:2])
                isPM = date[20:22]

                if isPM == 'PM':
                    if hour != 12:
                        hour += 12
                else:#AM
                    if hour == 12:
                        hour = 0
                features[lineNum][1] = hour
                features[lineNum][2] = month
                
            #0 = ID
            #2 = Date
            #4 = IUCR
            #11 = district 
            #13 = community area
            #14 = FBI Code
            #19 = lat
            #20 = long  
        
        return features

def removeZeroRows(features):
    nonZeroRows = []
    zed = np.zeros(8)
    for i, row in enumerate(features):
        if not np.allclose(row,zed):
            nonZeroRows.append(i)
    
    features2 = np.zeros((len(nonZeroRows),8))
    print('Counting done, now gather')
    for i,goodIndex in enumerate(nonZeroRows):
        features2[i] =  features[goodIndex]

    return features2


dataLines = clean_data('data/crimes.csv')
attribute_names = ['ID','Hour','Month','District','CommunityArea','Latitude','Longitude','FBICode']
print('REMOVING ZERO ROWS')
features = removeZeroRows(dataLines)
print('WRITING DATA')
write_data('data/cleanedData.csv',features,attribute_names)
print('DONE')







