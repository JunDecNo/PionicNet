import os
import numpy as np


def pd2list(text):
    if text is None:
        return []
    else:
        return np.fromstring(text.strip('[]'), dtype=np.float64, sep=' ')
    
def touch(path):
    if not os.path.exists(path):
        with open(path,'a') as file:
            pass
def getText(path):
    with open(path, 'r') as file:
        return file.readlines()
def makeDir(path, list):
    for i in list:
        os.mkdir(path + f'/{i}')
def appendText(path, text):
    with open(path, 'a') as file:
        file.write(text)
        
def calTheta(x, y):
    if np.all(x==0) or np.all(y==0):
        return 0
    return np.degrees(np.arccos(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))))




if __name__ == '__main__':
    print(calTheta([0, 0, 9], [0,1,0]))  