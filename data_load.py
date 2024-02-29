from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def load_data(dataPath, labelPath):
    predata = pd.read_excel(dataPath, header=None).values
    prelabel = pd.read_excel(labelPath, header=None).values
    predata = np.reshape(predata, (predata.shape[0], 30, 60))
    prelabel = np.reshape(prelabel, -1)
    trainData, testData, trainLabel, testLabel = train_test_split(predata, prelabel, train_size=0.5, random_state=907)
    return trainData, testData, trainLabel, testLabel