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
from sklearn.datasets import load_iris

def print_library_versions():
    print('Python: {}'.format(sys.version))
    print('scipy: {}'.format(scipy.__version__))
    print('numpy: {}'.format(numpy.__version__))
    print('matplotlib: {}'.format(matplotlib.__version__))
    print('pandas: {}'.format(pandas.__version__))
    print('sklearn: {}'.format(sklearn.__version__))

def load_dataset(url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"):
    """load dataset"""
    my_names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    local_dataset = read_csv(url, names=my_names) 
    return local_dataset

def load_dataset2():
    iris = load_iris()
    return iris

def load_clipboard():
    clp = pandas.read_clipboard(sep='\\s+')
    return clp

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

def split_dataset(dataset):
    """output from split X_train, X_validation, Y_train, Y_validation"""
    array = dataset.values
    X = array[:,0:4]
    Y = array[:,4]
    return train_test_split(X, Y, test_size=0.20, random_state=1)

def test_models(X_train, y_train):
    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))
    results = []
    names = []
    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
        #print('{}: {} ({})'.format(name, cv_results.mean(), cv_results.std()))
    pyplot.boxplot(results, labels=names)
    pyplot.title('Algorithm Comparison')
    pyplot.show()

def prediction_model(X_train, Y_train, X_validation, Y_validation):
    model = SVC(gamma='auto')
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)
    print(X_validation)
    print(predictions)
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))

def predict_iris_class(sepallength, sepalwidth, petallength, petalwidth):
    dataset = load_dataset()
    split = split_dataset(dataset)
    model = SVC(gamma='auto')
    model.fit(split[0], split[2])
    user_data = [sepallength, sepalwidth, petallength, petalwidth]
    iris_class = model.predict([user_data])
    return iris_class[0]
