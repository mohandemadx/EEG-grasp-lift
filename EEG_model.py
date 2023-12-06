# backend.py
from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score,roc_auc_score
import matplotlib.pyplot as plt
from sklearn import metrics
import glob
from scipy import signal
import seaborn as sn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import joblib


class EEGModel(QObject):
    resultReady = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        
    def detect_epilepsy(self, svc, data):
        
        y_predict = svc.predict(data)
        print(f'y_predict{y_predict}')
        
        if y_predict == 1:
            result = 'Closed'
        else:
            result = 'Open'
            
        return result