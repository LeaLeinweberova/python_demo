import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn

from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def print_library_versions():
    print('Python: {}'.format(sys.version))
    print('scipy: {}'.format(scipy.__version__))
    print('numpy: {}'.format(numpy.__version__))
    print('matplotlib: {}'.format(matplotlib.__version__))
    print('pandas: {}'.format(pandas.__version__))
    print('sklearn: {}'.format(sklearn.__version__))

def load_dataset(url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"):
    """load dataset"""
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = read_csv(url, names=names) 
    return dataset

def print_dataset(dataset):
    print('shape: {}'.format(dataset.shape))
    print(dataset.head(20))
    print(dataset.describe())
    print(dataset.groupby('class').size())

def plot_dataset(dataset, mykind='box', x=False, y=False):
    dataset.plot(kind=mykind, subplots=True, layout=(2,2), sharex=x, sharey=y)
    dataset.hist()
    scatter_matrix(dataset)
    pyplot.show()

def test_dataset(dataset):
    array = dataset.values
    X = array[:,0:4]
    Y = array[:,4]
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20, random_state=1)
    


    


