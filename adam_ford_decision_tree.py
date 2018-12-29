import numpy as np 
import util

from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def q1_train_test_split(X,y):
    '''
    Return a dictionary that contains 50 random train/test splits
    For the i-th split:
        splits[i][0]: X_train
        splits[i][1]: y_train
        splits[i][2]: X_test
        splits[i][3]: y_test
    '''
    random_seeds = [i for i in range(50)]
    splits = {}
    for i in range(50):
        # print random_seeds
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=random_seeds[i])
        splits[i] = []
        splits[i].append(X_train)
        splits[i].append(y_train)
        splits[i].append(X_test)
        splits[i].append(y_test)
    return splits


def decision_tree(X_train, y_train, X_test, y_test, max_depth=20):
    clf = tree.DecisionTreeClassifier(random_state=1, criterion= "entropy", max_depth=max_depth)
    clf = clf.fit(X_train, y_train)
    clf = clf.predict(X_test)
    accuracy = accuracy_score(clf, y_test)
    return accuracy

def cross_val(X_train,y_train):
    cv = KFold(n_splits = 5, shuffle=True, random_state=1)
    maxAccuracy = 0
    depth = 0
    for i in range(1,len(X_train)):
        accuracy = 0
        for train_index, test_index in cv.split(X_train):
            xtrain, xtest = X_train[train_index], X_train[test_index]
            ytrain, ytest = y_train[train_index], y_train[test_index]
            accuracy += decision_tree(xtrain,ytrain,xtest,ytest,max_depth=i)
        accuracy /= 5
        if accuracy > maxAccuracy:
            maxAccuracy = accuracy
            depth = i
            
    return depth

if __name__ == '__main__':
    X,y = util.load_data('dataset/Automobile_data.csv')
    '''----------------------q1----------------------'''
    splits = q1_train_test_split(X,y)
    avg_acc = 0.0
    for i in range(0,50):
        xtrain, ytrain, xtest, ytest = splits[i]
        avg_acc += decision_tree(xtrain, ytrain, xtest, ytest)
    avg_acc = avg_acc / 50
    
    #print splits[0][0].shape, splits[0][1].shape, splits[0][2].shape, splits[0][3].shape
    print 'Average accuracy is:', avg_acc

    '''----------------------q2----------------------'''
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=1)
    best_depth = cross_val(X_train,y_train)
    print 'Best max_depth found is:', best_depth

