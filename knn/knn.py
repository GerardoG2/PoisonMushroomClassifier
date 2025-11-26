import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer, OneHotEncoder, LabelEncoder
from sklearn.metrics import PrecisionRecallDisplay, accuracy_score, PredictionErrorDisplay
import numpy as np
import matplotlib.pyplot as plt

# Import raw data
raw_data = pd.read_csv("../Data/clean_data.csv")

# use l1 normalization on the numeric data
scaler = Normalizer(norm='l1')
numeric_data = raw_data.select_dtypes(include=[np.number])

# transform the numeric data
scaled_numeric_data = scaler.fit_transform(numeric_data)
scaled_numeric_df = pd.DataFrame(scaled_numeric_data, columns=numeric_data.columns)
scaled_numeric_df.describe()

# combine normalized numeric data with categorical data
categorical_data = raw_data.select_dtypes(include=['object']).reset_index(drop=True)
normalized_data = pd.concat([scaled_numeric_df, categorical_data], axis=1)
normalized_data.describe()

encoder = OneHotEncoder(sparse_output = False)

# categorical columns not including class
categorical_features = raw_data.select_dtypes(include=['object']).drop(columns=['class'])

# fit and transform the all categorical columns
encoded_features = encoder.fit_transform(raw_data[categorical_features.columns])

# encode target variable
le = LabelEncoder()
raw_data['class'] = le.fit_transform(raw_data['class'])

# combine encoded features with normalized numeric features
final_data = pd.concat([scaled_numeric_df, pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features.columns)), raw_data['class'].reset_index(drop=True)], axis=1)

X = final_data.drop(columns="class")
Y = final_data["class"]

# Split the dataset into an 80/20 train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=42)

# Use Cross-Validation Grid Search to test for optimal k value for a knn classifier using L2 norm Euclidean distance
# knn = KNeighborsClassifier(algorithm="brute")
# grid = {"n_neighbors": range(len(X_train))}

# grid_search = GridSearchCV(knn, grid)
# grid_search.fit(X_train, Y_train)
# optimal_k = grid_search.best_params_["n-neighbors"]
# print(optimal_k)

error_rates = []
k_range = range(1, len(X_train), 2)
errInc = 0
minErr = 1
optimalK = 1
# Use Elbow Method to find optimal k
# Stopping criteria: repeated decrease in accuracy
for k in k_range:
    knn = KNeighborsClassifier(algorithm="brute", n_neighbors=k)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    error = 1 - accuracy_score(Y_test, Y_pred)

    errInc = errInc + 1 if error_rates and error >= error_rates[-1] else 0
    error_rates.append(error)
    print(f"k={k}, err={error}, prevErr={error_rates[-1]}, errInc={errInc}")

    if error < minErr:
       optimalK = k
       minErr = error

    if errInc >= 10:
      print(f"Test stopped at k={k}")
      break

k_range = range(1, 2 * len(error_rates), 2)
print(len(k_range), len(error_rates))

plt.figure(figsize=(8, 6))
plt.plot(k_range, error_rates, marker='o')
plt.xlabel('K Value')
plt.ylabel('Error Rate')
plt.title('Error Rate vs K Value')
plt.show()
plt.saveFig("error-rate.png")