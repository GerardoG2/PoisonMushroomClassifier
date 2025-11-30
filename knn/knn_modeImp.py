
# KNN WITH MODE IMPUTATION

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split    

pd.set_option('display.max_columns', None)

df = pd.read_csv('../Data/raw_data.csv', sep=';')

columns_to_drop = ['stem-root','veil-type', 'veil-color',  'stem-surface', 'spore-print-color']
df = df.drop(columns=columns_to_drop)

from sklearn import set_config
from sklearn.preprocessing import LabelEncoder
set_config(transform_output="pandas") 


X = df.drop(columns='class')
y = df['class']

# Drop columns with more than 50% missing values
threshold = len(X) * 0.5
X = X.dropna(axis=1, thresh=threshold)

# label encode y 
le = LabelEncoder()
y = le.fit_transform(y)

# Encode features
numeric_features = X.select_dtypes(include='float64').columns.to_list()
categorical_features = X.select_dtypes(include='object').columns.to_list()

numeric_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(
    handle_unknown='ignore',
    sparse_output=False
)

transformer = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ]
)

X = transformer.fit_transform(X)

# Split training and test 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Impute missing values with mode
imputer = SimpleImputer(strategy="most_frequent", add_indicator=True)
imputer.fit(X_train)

X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)

print(X_train)

# 5NN Model
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