#####
#   BA    Amadou     16 187 314 
#   YING  Xu         18 205 032
#   ABOU Hamza       17 057 836
###
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class Metrics:

    def accuracy(self, model, y, x=None, label=None, pred=None):
        """
    	   label : Training | testing
    	"""
        if pred is None:
            print(label + ' accuracy', round(model.score(x, y) * 100, 2), " %")
        else:
            print('Testing accuracy after cross-validation ', round(metrics.accuracy_score(pred,y) * 100, 2), " %")


    def confusion_matrix(self, model, y, x=None, label=None, pred=None):
       if pred is None:
            plt.imshow(confusion_matrix(y, model.predict(x)), interpolation='nearest', cmap=plt.cm.Blues)

            for i in range(confusion_matrix(y, model.predict(x)).shape[0]):
                for j in range(confusion_matrix(y, model.predict(x)).shape[1]):
                    plt.text(j, i, str([['TN', 'FP'], ['FN', 'TP']][i][j]) + " = " + str(confusion_matrix(y, model.predict(x))[i, j]), ha="center", va="center")

            plt.title("CONFUSION MATRIX VISUALIZATION OF THE "+label.upper())

       else:
            plt.imshow(confusion_matrix(y, pred), interpolation='nearest', cmap=plt.cm.Blues)

            for i in range(confusion_matrix(y, pred).shape[0]):
                for j in range(confusion_matrix(y, pred).shape[1]):
                    plt.text(j, i, str([['TN', 'FP'], ['FN', 'TP']][i][j]) + " = " + str(confusion_matrix(y, pred)[i, j]), ha="center", va="center")

            plt.title("CONFUSION MATRIX VISUALIZATION AFTER VALIDATION ")

            plt.ylabel("True Values")
            plt.xlabel("Predicted Values")
            plt.xticks([])
            plt.yticks([])
            plt.show()
   

            
    

    