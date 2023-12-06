# main.py
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
import joblib
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from EEG_model import EEGModel
from PyQt5.uic import loadUiType
import pandas as pd
from os import path
import firebase_admin
from firebase_admin import credentials, db
import numpy as np


FORM_CLASS, _ = loadUiType(path.join(path.dirname(__file__), "designG&L.ui"))

class MainApp(QMainWindow, FORM_CLASS):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)
        
        # Firebase configuration
        cred = credentials.Certificate('C:/Users/memaa/Downloads/test-17bcc-firebase-adminsdk-r3ujd-94ca0c1166.json')
        firebase_admin.initialize_app(cred, {'databaseURL': 'https://test-17bcc-default-rtdb.firebaseio.com'})

        
        # Initialize the EEG model
        self.eeg_model = EEGModel()
        self.input = None
        self.eeg_data = None

        # Connect signals and slots
        self.importButton_2.clicked.connect(self.update_result)
        
        self.importButton_2.setEnabled(True)
        
    def export_to_firebase(self):
        data_to_export = self.detection_label.text()

        #Node of the Firebase
        ref = db.reference('Hand State:')
        ref.set(data_to_export)

        print('Data exported to Firebase:', data_to_export)

    def update_result(self):
        
        self.input = pd.read_csv('test.csv', index_col=None, header=0)
        
        # Random Row for Input
        random_row = self.input.sample(n=1)
        X = random_row.drop(columns = ['output'])
        y = random_row['output']
        print(X)
        print(y)
        
        loaded_svm_model = joblib.load('svm_model.pkl')
        # Update the GUI with the result
        
        result = self.eeg_model.detect_epilepsy(loaded_svm_model, X)
        self.detection_label.setText(result)
        self.export_to_firebase()


def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()