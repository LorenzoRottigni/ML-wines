from sklearn.datasets import load_wine
from utils import debug_DF
# Transform algoritmh used for scaling
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

dataset = load_wine()

print(dataset['DESCR'])

phenols_col = 4
magnesium_col = 7

X = dataset['data'][
    # : => take all rows
    :,
    # take phenols and magnesium columns
    [phenols_col, magnesium_col]
]

y = dataset['target']


""" Scaling data """
# debug_DF(X, 'wines_before.png')
# Scaling algorithm
scale = MinMaxScaler()

# Overwrite entry dataset with scaled data
s_X = scale.fit_transform(X)

# Build model with scaled data using KNN (K Nearest Neighbors) algorithm
s_model = KNeighborsClassifier(n_neighbors=3)
s_model.fit(s_X, y)
s_p = s_model.predict(X)

# Unscaled model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)
p = model.predict(X)

# debug_DF(X, 'wines_after.png')

s_acc = accuracy_score(y, s_p)
acc = accuracy_score(y, p)

print('Accuracy not scaled: ', s_acc)
print('Accuracy scaled: ', acc)
