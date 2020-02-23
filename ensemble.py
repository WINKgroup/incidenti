import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import sklearn.tree as tree
from sklearn.linear_model import Lasso
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

df = pd.read_csv('incidenti.csv', usecols=['GIORNOSETT', 'MESE2', 'TIME', 'Value'])
df.columns = ['weekday', 'month', 'year', 'value']
df['date'] = df['year'].astype('str') + '-' + df['month'].apply(lambda x: "%02d" % x)
df['date'] = pd.to_datetime(df['date'])
dfTotal = df.groupby(['date', 'weekday'], as_index=False).sum().drop(['year', 'month'], axis=1)
dfTotal = dfTotal.sort_values(by=['date', 'weekday'])

dfTotal[dfTotal['weekday']==1]['value'].plot(label='domenica')
dfTotal[dfTotal['weekday']==2]['value'].plot(label='lunedi')
plt.title('Andamento Incidenti')
plt.legend()

dfTotal['month'] = dfTotal['date'].dt.month
dfTotal['year'] = dfTotal['date'].dt.year

X = dfTotal[['year', 'month', 'weekday']].values
y = dfTotal['value'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

reg1 = tree.DecisionTreeRegressor()
#reg1.fit(X_train, y_train)

reg2 = Lasso()
#reg2.fit(X_train, y_train)

params = {'tree__max_depth': range(2, 20), 'lasso__alpha': np.linspace(.5, 2., num=10)}
ereg = VotingRegressor(estimators=[('tree', reg1), ('lasso', reg2)])
grid = GridSearchCV(estimator=ereg, param_grid=params, cv=2)
grid.fit(X_train, y_train)
print('Best params:', grid.best_params_)
print('Score:', grid.score(X_test, y_test))
