import numpy as np
from dataLoader import DataLoader
import math

class Predictor:
    def __init__(self):
        self.upMatrix = []
        self.downMatrix = []
        self.testSeqs = []
        self.getNewData()

    def getNewData(self, minD='2002-01-01', maxD='2023-01-01', count=10000, testP=0.2, minS=5, maxS=10, cutoff=0.6):
        # instantiate DataLoader object
        dl = DataLoader(minD, maxD, count,testP, minS, maxS, cutoff)
        # produce grids and test sequences
        dl.produceData(show=False)
        # store values
        self.upMatrix, self.downMatrix = dl.getMatrices()
        self.testSeqs = dl.getTestSeqs()
    
    # tests accuracy of model
    # loCut/hiCut: cutoff values for logOdds (if the probability is too weak we discard the prediction)
    def testModel(self, loCut=0, hiCut=0):
        predictions = np.empty(0)
        outcomes = np.empty(0)
        small = 0.00001
        odds = []
        
        # iterate through test sequences
        for id in self.testSeqs['SeqId'].values:
            seqRow = self.testSeqs[self.testSeqs['SeqId']==id]
            seq = seqRow['Sequence'].values[0]
            logOdds = 0

            # calculate logOdds for the sequence (sum of logOdds for each transition)
            for i in range(len(seq)-1):
                # calculate log value
                num = self.upMatrix.getEntry(seq[i], seq[i+1])
                den = self.downMatrix.getEntry(seq[i], seq[i+1])

                if num == 0 and den == 0:
                    pass
                else:
                    logOdds += math.log(max(num, small)/max(small, den))
            
            odds.append(logOdds)

            # cutoff value if necessary
            if logOdds > hiCut:
                predictions = np.append(predictions, 1)
                outcomes = np.append(outcomes, seqRow['Outcome'])
            elif logOdds < loCut:
                predictions = np.append(predictions, 0)
                outcomes = np.append(outcomes, seqRow['Outcome'])

        # calculate the number of discarded predictions
        removed = (len(self.testSeqs['SeqId'].values)-predictions.size)/len(self.testSeqs['SeqId'].values)
        print("removed " + '{:.1%}'.format(removed), end=' | ')
        
        # calculate, output and return model's accuracy for the test
        if predictions.size > 0:
            accuracy = np.count_nonzero((predictions-outcomes) == 0) / predictions.size
            print('{:.1%}'.format(accuracy) + f" accurate ({predictions.size})")
            return accuracy
        else:
            print("No predictions")
            return None

# FIX THIS
np.seterr(all='ignore')

# instantiate Predictor object
p = Predictor()

# set the number of tests to run
tests = 5

total = 0
for _ in range(tests):
    # test model
    p.getNewData()
    total += p.testModel()
# output average accuracy across all tests
print("Overall " + '{:.1%}'.format(total/tests) + " accurate")