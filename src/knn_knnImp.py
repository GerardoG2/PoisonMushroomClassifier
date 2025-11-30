
# KNN WITH KNN IMPUTATION 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split    
from sklearn.svm import SVC

pd.set_option('display.max_columns', None)

df = pd.read_csv('../Data/raw_data.csv', sep=';')

columns_to_drop = ['stem-root','veil-type', 'veil-color',  'stem-surface', 'spore-print-color']
df = df.drop(columns=columns_to_drop)
df.columns

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


class KNNCategoricalImputer(BaseEstimator, TransformerMixin):
    """
    Supervised KNN imputer for one or more categorical columns.

    - Assumes input X is a pandas DataFrame with original columns.
    - For each target column:
        * uses all other columns that have no missing values as predictors
        * does its own StandardScaler + OneHotEncoder internally
    """

    def __init__(self, target_cols, num_cols=None, n_neighbors=5, weights="distance"):
        # allow a single string or a list
        if isinstance(target_cols, str):
            target_cols = [target_cols]

        self.target_cols = target_cols
        self.num_cols = num_cols   # list of numeric columns to scale 
        self.n_neighbors = n_neighbors
        self.weights = weights

    def fit(self, X, y=None):
        df = X.copy()

        self.knn_models_ = {}
        self.feature_cols_ = {}

        for col in self.target_cols:
            # rows where the target is known
            mask = df[col].notna()

            # trainable columns: all except the target, and with no missing values on these rows
            _X_train_tmp = df.loc[mask].drop(columns=[col])
            train_columns = _X_train_tmp.columns[
                _X_train_tmp.isnull().sum() == 0
            ].tolist()

            X_train = df.loc[mask, train_columns]
            y_train = df.loc[mask, col]

            # numeric + categorical columns for THIS target
            if self.num_cols is None:
                # if not provided, infer numeric from dtypes
                num_cols = X_train.select_dtypes(include="number").columns.tolist()
            else:
                # keep only those num_cols that are actually in train_columns
                num_cols = [c for c in self.num_cols if c in train_columns]

            cat_cols = [c for c in train_columns if c not in num_cols + [col]]

            # preprocessor for KNN
            pre = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), num_cols),
                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
                ]
            )

            # KNN classifier pipeline
            clf = Pipeline(
                steps=[
                    ("pre", pre),
                    (
                        "knn",
                        KNeighborsClassifier(
                            n_neighbors=self.n_neighbors, weights=self.weights
                        ),
                    ),
                ]
            )

            # fit KNN for this target column
            clf.fit(X_train, y_train)

            # store model and the columns it expects
            self.knn_models_[col] = clf
            self.feature_cols_[col] = train_columns

        return self

    def transform(self, X):
        df = X.copy()

        for col in self.target_cols:
            mask_missing = df[col].isna()

            if mask_missing.any():
                feats = self.feature_cols_[col]
                X_missing = df.loc[mask_missing, feats]
                df.loc[mask_missing, col] = self.knn_models_[col].predict(X_missing)

        return df
from sklearn import set_config
from sklearn.preprocessing import LabelEncoder
set_config(transform_output="pandas") 


X = df.drop(columns='class')
y = df['class']

# label encode y 

le = LabelEncoder()
y = le.fit_transform(y)

numeric_features = X.select_dtypes(include='float64').columns.to_list()
categorical_features = X.select_dtypes(include='object').columns.to_list()

# Columns to impute
features_to_impute = ['cap-surface', 'gill-attachment', 'gill-spacing', 'ring-type']

numeric_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(
    handle_unknown='ignore',
    sparse_output=False  # 
)


knn_imputer = KNNCategoricalImputer(
    target_cols=features_to_impute,
    num_cols=numeric_features
)


column_trans = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ],
    verbose_feature_names_out=False
)


preprocess = Pipeline(steps=[
    ('knn_imputer', knn_imputer),      # impute missing values using KNN 
    ('column_transformer', column_trans)  
])




X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


preprocess.fit(X_train)

X_train = preprocess.transform(X_train)
X_test = preprocess.transform(X_test)

print(X_train)

# Use Cross-Validation Grid Search to test for optimal k value for a knn classifier using L2 norm Euclidean distance
# knn = KNeighborsClassifier(algorithm="brute")
# grid = {"n_neighbors": range(len(X_train))}

# grid_search = GridSearchCV(knn, grid)
# grid_search.fit(X_train, Y_train)
# optimal_k = grid_search.best_params_["n-neighbors"]
# print(optimal_k)

# error_rates = []
# k_range = range(1, len(X_train), 2)
# errInc = 0
# minErr = 1
# optimalK = 1
# Use Elbow Method to find optimal k
# Stopping criteria: repeated decrease in accuracy
# for k in k_range:
#     knn = KNeighborsClassifier(algorithm="brute", n_neighbors=k)
#     knn.fit(X_train, y_train)
#     Y_pred = knn.predict(X_test)
#     error = 1 - accuracy_score(y_test, Y_pred)

#     errInc = errInc + 1 if error_rates and error >= error_rates[-1] else 0
#     error_rates.append(error)
#     print(f"k={k}, err={error}, prevErr={error_rates[-1]}, errInc={errInc}")

#     if error < minErr:
#        optimalK = k
#        minErr = error

#     if errInc >= 10:
#       print(f"Test stopped at k={k}")
#       break

# k_range = range(1, 2 * len(error_rates), 2)
# print(len(k_range), len(error_rates))

# plt.figure(figsize=(8, 6))
# plt.plot(k_range, error_rates, marker='o')
# plt.xlabel('K Value')
# plt.ylabel('Error Rate')
# plt.title('Error Rate vs K Value')
# plt.show()

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
Y_pred = knn.predict(X_test)
print(classification_report(y_test, Y_pred))

cm_mixed = confusion_matrix(y_test, Y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_mixed, annot=True, fmt='d', cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('KNN Confusion Matrix')
plt.show()