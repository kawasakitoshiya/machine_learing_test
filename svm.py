import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score
)
from sklearn.svm import LinearSVC


def main():
    train = pd.read_csv('titanic/train.csv')
    train.Age = train.Age.fillna(train.Age.mean())
    le_sex = preprocessing.LabelEncoder()
    train.Sex = le_sex.fit_transform(train.Sex)
#    train.Sex = le_sex.inverse_transform(train.Sex)

    le_embarked = preprocessing.LabelEncoder()
    train.Embarked = le_embarked.fit_transform(train.Embarked)

    y = train['Survived']
    X = train[['Age', 'Sex', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Embarked']]

    estimator = LinearSVC(C=1.0)

    estimator.fit(X, y)
    py = estimator.predict(X)

    table = pd.crosstab(y, py)
    print '--table--'
    print table
    print '---------'

    print '--array--'
    print confusion_matrix(py, y)
    print '---------'

    print '--score--'
    print accuracy_score(py, y)
    print '---------'

if __name__ == '__main__':
    main()
