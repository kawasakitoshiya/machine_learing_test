import pandas as pd
from sklearn import linear_model


def main():
    train = pd.read_csv('titanic/train.csv')

    train.Age = train.Age.fillna(train.Age.mean())
    sex_dict = {
        'male': 1,
        'female': 0
    }
    for i, sex in enumerate(train.Sex):
        train.Sex[i] = sex_dict[sex]

    logiReg = linear_model.LogisticRegression()

    y = train['Survived']

    X = train[['Age', 'Sex']]
    logiReg.fit(X, y)
    print logiReg.coef_
    print logiReg.intercept_
    print logiReg.score(X, y)
    py = logiReg.predict(X)
    table = pd.crosstab(y, py)
    print '---'
    print table
    print '---'

if __name__ == '__main__':
    main()
