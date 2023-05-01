from sklearn.datasets import load_wine
from utils import debug_DF
# Transform algoritmh used for scaling
from sklearn.preprocessing import MinMaxScaler

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

debug_DF(X, 'wines_before.png')

# Scaling algorithm
scale = MinMaxScaler()

# Overwrite entry dataset with scaled data
X = scale.fit_transform(X)

debug_DF(X, 'wines_after.png')


