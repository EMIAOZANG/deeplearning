import pandas as pd
import sys
from sklearn.metrics import accuracy_score


try:
    fPath = sys.argv[1]
    y_pred = pd.read_csv(fPath)
    y_pred = list(y_pred['Prediction'])
    print type(y_pred)
    y_true = []
    for i in xrange(1,11):
        y_true += [i] * 800

    print "Test accuracy: ", accuracy_score(y_true, y_pred)



except:
    sys.exit()


