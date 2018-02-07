import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import random
from scipy.io import loadmat

def main():
    d=61188
    n_train = 11269.0
    n_test = 7505
    news = loadmat("news.mat")
    
    print "loaded"
    
    data = news['data']
    labels = news['labels'].flatten()
    testData = news['testdata']
    testLabels = news['testlabels'].flatten()
    classpriors = []
    
    mu = np.asfarray(np.empty((20,d)))
    for y in range(1,21):
        partition = data[np.where(news['labels']==y)[0]]
        n_y = partition.shape[0]
        classpriors.append(n_y/n_train)
        num = partition.sum(axis = 0)+1
        den = 2+n_y
        mu_y = num/(float)(den)
        mu[y-1]=mu_y
    trainingPreds = classify(classpriors, mu, data)
    error = np.count_nonzero(labels != trainingPreds)
    print "Training Data Error: " + str(error/(float)(n_train))
    testPreds = classify(classpriors, mu, testData)
    error = np.count_nonzero(testLabels != testPreds)
    print "Test Data Error: " + str(error/float(n_test))
def classify(classpriors, mu, test):
    logCP = np.log(classpriors)
    logMu = np.log(mu)
    log1Mu= np.log(1-mu)
    term1=test.dot(logMu.T)
    term2 = test.dot(log1Mu.T)
    P = term1-term2+term3+np.log(classpriors)
    return np.argmax(P,axis=1)+1


if __name__ == "__main__":
    main()

