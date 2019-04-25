import numpy as np
import UCT_mod
from random import *
#import UCT
import pandas as pn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def playGames(numGames):
    #  Function to gather sample data
    for i in range(0, numGames):
        UCT_mod.UCTPlayGame()
    return

gameData = pn.read_csv("gameData.csv", sep=',')  # Read learning data from file
#  print(len(gameData))
#  print(type(gameData))
#  for i in gameData:
#    print(i)
features = gameData.values[:,0:9]
labels = gameData.win.values
# convert both types to int (labels are boolean, simpler to work as integer)
features = features.astype(np.int)
labels = labels.astype(np.int)

def manualReview():
    # Review random datasets
    randomSeed = 10
    flaggedSets = 0
    for i in range(0,len(labels)):
        #  Manually review a dataset
        if randint(1, randomSeed) == 1:
            if labels[i] == 1:
                print("Data predicts a win for player 1. Is this correct?:")
            else:
                print("Data predicts a lose for player 1 or a draw. Is this correct?:")
            # Print the game board in correct 3 row layout
            print(features[i, 0:3])
            print(features[i, 3:6])
            print(features[i, 6:9])
#  print(features)
featuresTraining, featuresTest, labelsTraining, labelsTesting = train_test_split(features, labels, random_state=3, shuffle=True)
#gameData.corr()

#k_fold = KFold(n_splits=10)
#clf = RandomForestClassifier(n_estimators = 1000,max_depth = 9)
#  scoreTree = crossValScore(clf, gameData.data, gameData.target, cv=k_fold, n_jobs=-1)
#  print('Average accuracy:', np.mean(scoreTree))

clf = DecisionTreeClassifier(criterion='gini', presort=True)
clf.fit(featuresTraining, labelsTraining)
decTreeScore = clf.score(featuresTest, labelsTesting)
#  print(decTreeScore) # Returns 0.992 on current dataset
def accuracyTest():
    # Now, verify the tree's accuracy on the real data
    accuracy = 0
    print("Testing decision tree accuracy...")
    for i in range(0, len(labels)):
        #  print(features[i, :])
        if clf.predict([features[i, :]]) == labels[i]:
            #  print(clf.predict([features[i, :]]))
            #  print(labels[i])
            accuracy += 1
        #  If the prediction was incorrect, report it
        else:
            print("Incorrect result:", i, features[i, :], "Predicted:", clf.predict([features[i, :]]), "Real outcome:", labels[i])
    accuracy = (accuracy / len(labels)) * 100
    print("Accuracy is", accuracy, "%")
    return
#tmpp = clf.predict([[2,1,2,1,2,1,2,1,2]])
#print(tmpp)
accuracyTest()

print("Done")