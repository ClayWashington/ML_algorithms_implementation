# Based on the algorithm as defined at https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html

import numpy as np
import math

class MyMultinomialNB:
    
    def __init__(self, X, y, alpha=1.0):
        self.classes = np.unique(y)
        self.priors = MyMultinomialNB.calculate_priors(y, self.classes)
        self.conditionals = MyMultinomialNB.calculate_conditionals(X, y, self.classes, alpha)


    def calculate_conditionals(X, y, classes, alpha):
        class_matricies = []
        conditionals = np.zeros((len(classes), X.shape[1]))
        counts_per_class = []
        counts_per_term = []
        class_indicies = MyMultinomialNB.get_indicies(y, classes)
        for c in class_indicies:
            class_matricies.append(X[c])
            counts_per_class.append(np.sum(class_matricies[-1]))
            counts_per_term.append(np.sum(class_matricies[-1], 0))
        counts_per_term = np.array(counts_per_term) # counts per term per class
        counts_per_class = np.array(counts_per_class) # total number of terms in each class
        for i in range(0, len(classes)):
            # Apply Laplace smoothing
            conditionals[i,:] = (counts_per_term[i,:] + alpha) / (counts_per_class[i] + X.shape[1])
        return conditionals


    def get_indicies(y, classes):
        class_indicies = []
        for c in classes:
            class_indicies.append(np.where(y==c)[0])
        return class_indicies


    def calculate_priors_d(y, classes):
        probC = dict.fromkeys(classes, 0)
        numC = dict.fromkeys(classes, 0) # Number of documents in class
        num = len(y)
        for c in y:
            numC[c] = numC[c] + 1
        for key, val in probC.items():
            probC[key] = numC[key] / num
        return probC


    def calculate_priors(y, classes):
        probC = np.zeros((len(classes),))
        numC = dict.fromkeys(classes, 0) # Number of documents in class
        num = len(y)
        for c in y:
            numC[c] = numC[c] + 1
        for index, c in enumerate(classes):
            probC[index] = numC[c] / num
        return probC

		
    def predict(self, doc):
        return np.argmax(self.make_predictions(doc))

		
    def predict_proba(self, doc):
        predictions = self.make_predictions(doc)
        predictions = math.e ** predictions
        total = predictions.sum()
        if total == 0:
            predictions = predictions + .001
            total = predictions.sum()
        predictions = predictions / predictions.sum()
        return predictions

		
    def make_predictions(self, doc):
        predictions = np.zeros(len(self.classes)) # A prediction for each class
        for i in range(0, len(self.classes)):
            predictions[i] = math.log(self.priors[i])
            indicies = np.where(doc > 0)[0]
            for index in indicies:
                predictions[i] = predictions[i] + math.log(self.conditionals[i, index])
        return predictions