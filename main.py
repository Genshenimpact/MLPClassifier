import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
data = pd.read_csv('adm_data.csv')
data['label'] = [1 if i > 0.7 else 0 for i in data['Chance of Admit ']]
clf = MLPClassifier()
X = data[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']]
y = data['label']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)
clf.fit(X_train,y_train)
score1 = clf.score(X_test,y_test)
print(score1)
param_grid = {'hidden_layer_sizes': [(10,7), (10, 10)],
              'activation': ['logistic', 'relu', 'identity', 'tanh'],
              'solver': ['sgd', 'adam', 'lbfgs'],
              'alpha': [0.000001, 0.00005],
              'learning_rate': ['constant', 'adaptive']}

grid = GridSearchCV(clf, param_grid,n_jobs = 8)
grid.fit(X_train, y_train)

print("最佳参数: ", grid.best_params_)
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Test set score: {:.2f}".format(grid.score(X_test, y_test)))

