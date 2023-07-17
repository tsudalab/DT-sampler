from DT_sampler import DT_sampler
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split

data = pd.read_csv('./data/mouse.csv',delimiter=',', header=None).to_numpy()

X = data[:,:-1]
y = data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 50, random_state=42)

sk = SelectKBest(chi2, k=10).fit(X_train, y_train)
X_train = sk.transform(X_train)
X_test = sk.transform(X_test)

dt_sampler = DT_sampler(X_train, y_train, 9, 45, "./cnf/test.cnf")
dt_sampler.run(50, method = "unigen", sample_seed = 0)

y_predicted_train = dt_sampler.predict(X_train)
print("training acc: ", sum(y_predicted_train == y_train)/len(y_train))

y_predicted = dt_sampler.predict(X_test)
print("test acc: ", sum(y_predicted == y_test)/len(y_test))

f_prob = dt_sampler.feature_prob()
print("The emergence probability of features:", f_prob)