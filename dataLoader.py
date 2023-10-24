import pandas as pd
import numpy as np
from random import randint

# matrix data structure designed to store transition probabilities between event patterns
class LMHMatrix:
    def __init__(self):
        # actual matrix in which values are stored
        self.matrix = np.zeros((27,27))

    # calculates the index in the matrix for a given event pattern
    # k: the event pattern
    def keyToIndex(self, k):
        kToI = {"L":0, "M":1, "H":2}
        return 9*kToI[k[0]] + 3*kToI[k[1]] + kToI[k[2]]
    
    # get value of a transition
    # r: event pattern transitioned from
    # c: event pattern transitioned to
    def getEntry(self, r, c):
        return self.matrix[self.keyToIndex(r)][self.keyToIndex(c)]
    
    # increment value of a transition
    # r: event pattern transitioned from
    # c: event pattern transitioned to
    def incEntry(self, r, c):
        self.matrix[self.keyToIndex(r)][self.keyToIndex(c)] += 1

    # replaces each entry with its proportion of the sum of values in its row
    # used to replace count values with probabilities
    def collapseValues(self):
        for r in range(len(self.matrix)):
            total = sum(self.matrix[r])
            if total:
                for c in range(len(self.matrix[r])):
                    self.matrix[r][c] = self.matrix[r][c]/total
    
    # returns a string that summarises the matrix
    # used to output the matrix for debugging
    def out(self):
        out = "["
        for i in range(min(4, len(self.matrix))):
            out += str(self.matrix[i][:min(3, len(self.matrix[i]))]) + ',\n'
        out = out[:-1]+"\n...]"
        print(out)

# class used for producing Markov transition grids and test sequences
class DataLoader:
    def __init__(self, minD, maxD, count, testP, minS, maxS, cutoff):
        # matrix containing probability of bull outcome for a given transition
        self.upMatrix = LMHMatrix()
        # matrix containing probability of bear outcome for a given transition
        self.downMatrix = LMHMatrix()
        # sequences used to test the model
        self.testSequences = []

        self.minDate = minD     # start date for reading stock data
        self.maxDate = maxD     # end date for reading stock data
        self.count = count      # total number of sequences to produce
        self.testProp = testP   # proportion of sequences to use for testing (rather than training)
        self.minSize = minS     # minimum sequence size
        self.maxSize = maxS     # maximum sequence size
        self.cutoff = cutoff    # cutoff value for outcomes (to remove weak outcomes)

    # produces transition grids and test sequences
    # show: determines whether process is shown to user
    def produceData(self, show=False):
        # DEFINE FUNCTIONS

        # calculates and formats percentage
        def asPercentage(n, total):
            if total == 0: return "NaN"
            return '{:.1%}'.format((total-n)/total)

        # loads file into pandas
        # fileName: name of CSV file storing stock data
        def loadStockData(fileName):
            # convert from CSV to pandas DataFrame
            df = pd.read_csv(fileName)

            # convert Date column to datetime data
            df['Date'] = pd.to_datetime(df['Date'])

            # remove values outside start and end dates
            df['Date'] = df['Date'].clip(lower=self.minDate, upper=self.maxDate)
            return df

        # produces sequences of % changes
        # df: DataFrame containing stock data
        # count: number of sequences to produce
        # minSize/maxSize: min/max sequence length
        # outcomeCutoff: cutoff value for absolute percentage volume change
        def parseStockData(df, count, minSize, maxSize, outcomeCutoff):
            # ensures maxSize does not exceed maximum possible sequence length
            maxSize = min(maxSize, len(df)-1)

            # holds outcome for each sequence
            idToOutcome = {}
            # holds date for each sequence
            idToDate = {}

            # holds data for each transition           
            seqIdFrame = []
            closeGapFrame = []
            volGapFrame = []
            dailyChangeFrame = []
            
            # take count samples
            n = 0
            while n < count:
                # take random index
                size = randint(minSize, maxSize)
                i = randint(2, len(df)-size-2)
                # find data
                subset = df.iloc[i-1:i+size]

                # returns percentage changes of each transition in a sequence
                # dataFrame is a sequence of events (only a single column)
                def formatPctChange(dataFrame):
                    return dataFrame.pct_change().to_numpy()[1:]
                
                # calculate outcome
                volChange = (df['Volume'][i+size+1] - df['Volume'][i+size]) / max(df['Volume'][i+size], 0.00000001)
                # ignore if volume change is too small
                if abs(volChange) < outcomeCutoff:
                    continue

                # record percentage changes close gap, volume gap, daily change for each transition in a sequence
                seqIdFrame += [n] * (len(subset)-1)
                closeGapFrame = np.concatenate( (closeGapFrame, formatPctChange(subset['Close'])) )
                volGapFrame = np.concatenate( (volGapFrame, formatPctChange(subset['Volume'])) )
                dailyChangeFrame = np.concatenate( (dailyChangeFrame, ((subset['Close'] - subset['Open']) / subset['Open'])[1:]) )

                # store sequence outcome and date
                idToOutcome[n] = volChange
                idToDate[n] = subset['Date'].values[-1]

                n += 1

            return pd.DataFrame({
                        'SeqId': seqIdFrame,
                        'CloseGap': closeGapFrame,
                        'VolumeGap': volGapFrame,
                        'DailyChange': dailyChangeFrame,
                }), idToOutcome, idToDate

        # categorise events into (L)ow, (M)edium, and (H)igh changes
        # returns upMatrix[in][out], downMatrix[in][out]
        def categoriseStockData(records):
            # uses pd.qcut to put %change values into bins (L,M,H)
            def categoriseAttribute(attribute):
                return pd.qcut(records[attribute], 3, labels=['L', 'M', 'H'])
            
            # categorises each column
            newRecords = pd.DataFrame({
                'SeqId': records['SeqId'],
                'CloseGap': categoriseAttribute('CloseGap'),
                'VolumeGap': categoriseAttribute('VolumeGap'),
                'DailyChange': categoriseAttribute('DailyChange'),
            })

            # concatenates each events into single event pattern
            newRecords['EventPattern'] = newRecords['CloseGap'].astype(str) + newRecords['VolumeGap'].astype(str) + newRecords['DailyChange'].astype(str)

            # return patterns
            return pd.DataFrame({
                'SeqId': newRecords['SeqId'],
                'EventPattern': newRecords['EventPattern']
            })
            
        # compresses sequences into data single entries
        def buildSequences(newRecords, idToOutcome, idToDate):
            seqIdFrame = []
            dateFrame = []
            sequenceFrame = []
            outcomeFrame = []

            # for each outcome
            for id in idToOutcome.keys():
                # find transitions for the sequence
                subset = newRecords.loc[newRecords['SeqId'] == id]
                # store sequence ID
                seqIdFrame.append(id)
                # store date
                dateFrame.append(idToDate[id])
                # create sequences of event patterns
                sequenceFrame.append(subset['EventPattern'].to_numpy())
                # simplify outcome (1,0)
                outcomeFrame.append(int(idToOutcome[id]>0))

            return pd.DataFrame({
                "SeqId": seqIdFrame,
                "Date": dateFrame,
                "Sequence": sequenceFrame,
                "Outcome": outcomeFrame
            })
        
        # builds and returns transition grids
        # sequences is training sequences for the model
        def buildTransitionGrids(sequences):
            upMatrix = LMHMatrix()
            downMatrix = LMHMatrix()

            # for each sequence
            for _, seq in sequences.iterrows():
                # current sequence
                curr = seq['Sequence']

                # for each transition
                for i in range(len(curr)-1):
                    # record bull outcome
                    if seq['Outcome'] == 1:
                        upMatrix.incEntry(curr[i], curr[i+1])

                    # record bear outcome
                    else:
                        downMatrix.incEntry(curr[i], curr[i+1])
            
            # change transition count values to transition probability values
            upMatrix.collapseValues()
            downMatrix.collapseValues()

            return upMatrix, downMatrix

        # CALL FUNCTIONS
        if show: 
            print("Loading stock data...")

        df = loadStockData("stockData.csv")

        if show:
            print("Imported file:")
            print(df)

            print("\nParsing stock data...")
        
        records, idToOutcome, idToDate = parseStockData(df, self.count, self.minSize, self.maxSize, self.cutoff)

        if show:
            print("Records:")
            print(records)

            print("\nCleaning records...")
        
        newRecords = categoriseStockData(records)

        if show:
            print("New Records:")
            print(newRecords)

            print("\nBuilding sequences...")

        sequences = buildSequences(newRecords, idToOutcome, idToDate)

        # split sequences into testing and training 
        testCount = int(len(sequences) * self.testProp)
        self.testSequences = sequences.tail(testCount)
        sequences = sequences.head(len(sequences)-testCount)

        if show:
            print("Training sequences:")
            print(sequences)
            print(asPercentage(sequences[sequences['Outcome']==1].shape[0], sequences['Outcome'].shape[0]), end="")
            print(" of values are positive")

            print("Test sequences:")
            print(self.testSequences)

            print("\nBuilding transition grid...")

        # produce and store transition grids
        self.upMatrix, self.downMatrix = buildTransitionGrids(sequences)

        if show:
            print("Up matrix:")
            self.upMatrix.out()
            print("Down matrix:")
            self.downMatrix.out()

            print("Complete")

    # returns transition matrices
    def getMatrices(self):
        return self.upMatrix, self.downMatrix
    
    # returns test sequences
    def getTestSeqs(self):
        return self.testSequences

# dl = DataLoader(minD='2010-01-01', maxD='2023-01-01', count=5000, testP=0.2, minS=5, maxS=10, cutoff=0.5)
# dl.produceData(show=True)