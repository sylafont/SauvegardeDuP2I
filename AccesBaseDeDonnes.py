from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from dependency import pickle
import os


def Load_Data():
  (X_train, y_train), (X_test, y_test) = mnist.load_data()
  y_train = y_train.reshape(60000,)
  y_test = y_test.reshape(10000,)
  X_train = X_train.reshape(60000, 784)
  X_test = X_test.reshape(10000, 784)
  X_train = X_train.astype('float32')
  X_test = X_test.astype('float32')
  X_train /= 255
  X_test /= 255
  return X_train, y_train, X_test, y_test

def Save_Data_Base(dataBaseName):
  X_train, y_train, X_test, y_test = Load_Data()
  try:
    with open("DataBase/"+ dataBaseName+ ".pickle", "wb") as f:
      pickle.dump((X_train, y_train, X_test, y_test), f, protocol=pickle.HIGHEST_PROTOCOL)#Warning with highest protocol peut être incompatible avec certaines version de python
  except Exception as ex:
    print("Error during pickling object (Possibly unsupported):", ex)

 
def Load_Data_From_Pickle(dataBaseName) :
  try:
    with open("DataBase/"+ dataBaseName+ ".pickle", "rb") as f:
      return pickle.load(f)
  except Exception as ex:
    print("Error during unpickling object (Possibly unsupported):", ex)

def IsInFolder(fileName, folderName):
  for path, subdirs, files in os.walk(folderName):
    for name in files:
        if name == fileName:
            return True
  return False
