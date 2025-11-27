"""
Mushroom Data Preprocessing Module

This module provides comprehensive preprocessing functions for mushroom classification data,
including KNN imputation, standardization, and encoding for both regular train/test splits
and cross-validation scenarios.
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer


def knn_impute_and_standardize(df, target, scaler=None, fit_scaler=False):
    """
    Enhanced KNN imputation that also handles numeric standardization
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with missing values
    target : str
        Column name to impute
    scaler : StandardScaler, optional
        Pre-fitted scaler for numeric columns
    fit_scaler : bool, default=False
        Whether to fit a new scaler (not used in current implementation)
        
    Returns:
    --------
    tuple: (imputed_dataframe, scaler)
    """
    # Define numeric columns explicitly within function
    numeric_cols = ['cap-diameter', 'stem-height', 'stem-width']
    
    mask = df[target].notna()
    
    # Get complete columns for training
    _X_train_tmp = df.loc[mask].drop(columns=[target])
    train_columns = _X_train_tmp.columns[_X_train_tmp.isnull().sum() == 0].tolist()
    
    # Prepare training data
    X_train = df.loc[mask, train_columns].copy()
    y_train = df.loc[mask, target]
    X_missing = df.loc[~mask, train_columns].copy()
    
    # Separate numeric and categorical columns
    available_num_cols = [col for col in numeric_cols if col in train_columns]
    available_cat_cols = [col for col in train_columns if col not in available_num_cols]
    
    # Standardize numeric columns - Only if we have numeric columns and a scaler
    if available_num_cols and scaler is not None:
        X_train[available_num_cols] = scaler.transform(X_train[available_num_cols])
        
        if not X_missing.empty:
            X_missing[available_num_cols] = scaler.transform(X_missing[available_num_cols])
    
    # Create preprocessor for KNN
    preprocessor_steps = []
    if available_num_cols:
        preprocessor_steps.append(("num", "passthrough", available_num_cols))
    if available_cat_cols:
        preprocessor_steps.append(("cat", OneHotEncoder(handle_unknown="ignore"), available_cat_cols))
    
    if preprocessor_steps:
        preprocessor = ColumnTransformer(preprocessor_steps)
        
        # KNN pipeline
        knn_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("knn", KNeighborsClassifier(n_neighbors=5, weights="distance"))
        ])
        
        # Fit and predict
        knn_pipeline.fit(X_train, y_train)
        if not X_missing.empty:
            df.loc[~mask, target] = knn_pipeline.predict(X_missing)
    
    return df, scaler


def preprocess_mushroom_data(train_df, test_df=None, fit_encoders=True, encoders=None):
    """
    Complete preprocessing pipeline for mushroom data.
    
    Parameters:
    -----------
    train_df : pandas.DataFrame
        Training dataframe
    test_df : pandas.DataFrame, optional
        Test dataframe. If None, only train_df is processed (useful for CV)
    fit_encoders : bool, default=True
        Whether to fit new encoders or use provided ones
    encoders : dict, optional
        Pre-fitted encoders for cross-validation. Should contain:
        {'scaler': StandardScaler, 'onehot': OneHotEncoder, 'label': LabelEncoder}
    
    Returns:
    --------
    If test_df is provided:
        tuple: (processed_train_df, processed_test_df, encoders_dict)
    If test_df is None:
        tuple: (processed_train_df, encoders_dict)
    """
    # Define column types
    numeric_cols = ['cap-diameter', 'stem-height', 'stem-width']
    categorical_cols_with_nans = ['cap-surface', 'gill-attachment', 'gill-spacing', 'ring-type']
    
    # Initialize encoders dictionary
    if encoders is None:
        encoders = {}
    
    # Copy dataframes to avoid modifying originals
    processed_train = train_df.copy()
    processed_test = test_df.copy() if test_df is not None else None
    
    # Step 1: Standardize numeric columns first (before imputation)
    if fit_encoders:
        scaler = StandardScaler()
        processed_train[numeric_cols] = scaler.fit_transform(processed_train[numeric_cols])
        encoders['scaler'] = scaler
    else:
        scaler = encoders.get('scaler')
        if scaler:
            processed_train[numeric_cols] = scaler.transform(processed_train[numeric_cols])
    
    # Apply same scaling to test set
    if processed_test is not None and scaler:
        processed_test[numeric_cols] = scaler.transform(processed_test[numeric_cols])
    
    # Step 2: Impute missing categorical values - FIXED: Pass the scaler
    for col in categorical_cols_with_nans:
        if col in processed_train.columns:
            processed_train, _ = knn_impute_and_standardize(processed_train, col, scaler=scaler)
            
            if processed_test is not None and col in processed_test.columns:
                processed_test, _ = knn_impute_and_standardize(processed_test, col, scaler=scaler)
    
    # Step 3: One-hot encode categorical features
    # Get all categorical columns (excluding class)
    categorical_features = [col for col in processed_train.columns 
                          if col not in numeric_cols + ['class'] and processed_train[col].dtype == 'object']
    
    if categorical_features:
        if fit_encoders:
            onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            train_categorical_encoded = onehot_encoder.fit_transform(processed_train[categorical_features])
            encoders['onehot'] = onehot_encoder
        else:
            onehot_encoder = encoders.get('onehot')
            train_categorical_encoded = onehot_encoder.transform(processed_train[categorical_features])
        
        # Get feature names
        feature_names = onehot_encoder.get_feature_names_out(categorical_features)
        
        # Create encoded dataframes
        train_encoded_df = pd.DataFrame(train_categorical_encoded, 
                                       columns=feature_names, 
                                       index=processed_train.index)
        
        # Combine numeric + encoded categorical
        train_features = pd.concat([
            processed_train[numeric_cols],
            train_encoded_df
        ], axis=1)
        
        # Process test set
        if processed_test is not None:
            test_categorical_encoded = onehot_encoder.transform(processed_test[categorical_features])
            test_encoded_df = pd.DataFrame(test_categorical_encoded,
                                         columns=feature_names,
                                         index=processed_test.index)
            test_features = pd.concat([
                processed_test[numeric_cols],
                test_encoded_df
            ], axis=1)
        else:
            test_features = None
    else:
        # No categorical features, just use numeric
        train_features = processed_train[numeric_cols]
        test_features = processed_test[numeric_cols] if processed_test is not None else None
    
    # Step 4: Label encode target variable
    if 'class' in processed_train.columns:
        if fit_encoders:
            label_encoder = LabelEncoder()
            train_labels = label_encoder.fit_transform(processed_train['class'])
            encoders['label'] = label_encoder
        else:
            label_encoder = encoders.get('label')
            train_labels = label_encoder.transform(processed_train['class'])
        
        # Process test labels
        if processed_test is not None and 'class' in processed_test.columns:
            test_labels = label_encoder.transform(processed_test['class'])
        else:
            test_labels = None
    else:
        train_labels = None
        test_labels = None
    
    # Combine features and labels
    if train_labels is not None:
        final_train = train_features.copy()
        final_train['class'] = train_labels
    else:
        final_train = train_features
    
    if test_features is not None:
        if test_labels is not None:
            final_test = test_features.copy()
            final_test['class'] = test_labels
        else:
            final_test = test_features
    else:
        final_test = None
    
    # Return results
    if final_test is not None:
        return final_train, final_test, encoders
    else:
        return final_train, encoders

