import numpy as np


def pd2list(text):
    if text is None:
        return []
    else:
        return np.fromstring(text.strip('[]'), dtype=np.float64, sep=' ')
    
def touch(path,mode='w'):
    if mode=='w':
        with open(path,'w') as file:
            file.write('')
    else:
        with open(path,'a') as file:
            pass
        