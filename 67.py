import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import re
import csv
# function for dataset1
def fixDataframe(df):
    for i in range(0, 13):
        df['f'+str(i+1)] = df['input_emoticon'].str.slice(i, i+1)
    df.drop('input_emoticon', axis=1, inplace=True)
    return df

# Function to encode each row based on the emoji-to-index mapping
def encode_row(row, emoji_to_index, num_emojis):
    encoded = np.zeros(num_emojis, dtype=int)
    unique_emojis = pd.unique(row)
    for emoji in unique_emojis:
        if emoji in emoji_to_index:
            encoded[emoji_to_index[emoji]] = 1
    return encoded

# Encode the dataset1
def encode_dataset(df, emoji_to_index_f1_to_f8, emoji_to_index_f9_to_f13):
    X_f1_to_f8 = np.array([encode_row(row, emoji_to_index_f1_to_f8, 213) for _, row in df.iloc[:, :8].iterrows()])
    X_f9_to_f13 = np.array([encode_row(row, emoji_to_index_f9_to_f13, 91) for _, row in df.iloc[:, 8:13].iterrows()])
    return np.hstack([X_f1_to_f8, X_f9_to_f13])




# Function for dataset3

def process_dataframes(dftrain, dfvalid, dftest,subsequences_to_remove):
    # Define overlapping strings and the max_replacements for each subsequence
    overlapping_strings = {'26284', '42284', '61422', '61464', '28422', '46422', '159614', '1543614','42262'}
    max_replacements = {
        '15436': 1, '1596': 2, '000': 1, '464': 1,
        '614': 2, '262': 2, '422': 1, '284': 1
    }

    def remove_subsequences(string, limits):
        # Loop through each subsequence and remove it up to the allowed number of times
        for sub, max_count in limits.items():
            string = re.sub(sub, '.', string, count=max_count)
        return string
    
    def removedd_subsequences(string, limits):
        # Loop through each subsequence and remove it up to the allowed number of times
        for sub, max_count in limits.items():
            string = re.sub(sub, '.', string, count=max_count)
        return string

    def process_string(input_str):
        # First check if any overlapping string is present
        overlapping_match = None
        for overlap in overlapping_strings:
            match = re.search(overlap, input_str)
            if match:
                overlapping_match = (overlap, match.start(), match.end())
                break
        
        # If an overlapping substring is found, mark its position
        if overlapping_match:
            overlap_str, start, end = overlapping_match
            
            # Remove the overlapping substring temporarily
            remaining_str = input_str[:start] + input_str[end:]
            
            # Apply the original logic to the remaining string
            processed_str = remove_subsequences(remaining_str, max_replacements.copy())
            processed_str = processed_str.replace('.', '')

            # Now, process the overlapping substring based on remaining limits
            # We remove the part of the overlap whose limit has not been exhausted
            overlap_limits = max_replacements.copy()

            # Adjust the limits based on what was already removed from the processed string
            for sub in overlap_limits:
                count_removed = len(re.findall(sub, remaining_str))  # Count how many were removed
                overlap_limits[sub] = max(0, overlap_limits[sub] - count_removed)

            # Apply the remaining limits to the overlapping substring
            processed_overlap = remove_subsequences(overlap_str, overlap_limits)
            processed_overlap = processed_overlap.replace('.','')

            # Reconstruct the final processed string
            final_str = processed_str[:start] + processed_overlap + processed_str[start:]
        else:
            # If no overlapping substring, apply the original removal logic
            final_str = removedd_subsequences(input_str, max_replacements.copy())
            final_str = final_str.replace('.', '')

        return final_str

    def process_df(df):
        new_df = df.copy()
        new_df['processed_str'] = new_df['input_str'].apply(process_string)
        
        # Find rows where processed strings do not have length 13
        non_matching_df = new_df[new_df['processed_str'].str.len() != 13]
        
        # Keep only rows where processed strings have length 13
        matching_df = new_df[new_df['processed_str'].str.len() == 13]
        
        # Drop the original 'input_str' column and rename the processed column to 'input_str'
        matching_df = matching_df.drop(columns=['input_str']).rename(columns={'processed_str': 'input_str'})
        
        return matching_df, non_matching_df

    # Process train and validation datasets
    dftrain2, dftrain_non_matching = process_df(dftrain)
    dfvalid2, dfvalid_non_matching = process_df(dfvalid)
    dftest2, dftest_non_matching = process_df(dftest)
    
    return dftrain2, dftrain_non_matching, dfvalid2, dfvalid_non_matching, dftest2,dftest_non_matching



def process_dataframes2(dftrain, dfvalid,dftest ,subsequences_to_remove):
    # Define overlapping strings and the max_replacements for each subsequence
    overlapping_strings = {'26284', '42284', '61422', '61464', '28422', '46422', '159614', '1543614','42262'}
    max_replacements = {
        '15436': 1, '1596': 2, '000': 1, '464': 1,
        '614': 2, '262': 2, '422': 1, '284': 1
    }

    def remove_subsequences(string, limits):
        # Loop through each subsequence and remove it up to the allowed number of times
        for sub, max_count in limits.items():
            string = re.sub(sub, '.', string, count=max_count)
        return string
    
    def removedd_subsequences(string, limits):
        # Loop through each subsequence and remove it up to the allowed number of times
        for sub, max_count in limits.items():
            string = re.sub(sub, '.', string, count=max_count)
        return string

    def process_string(input_str):
        # First check if any overlapping string is present
        overlapping_match = None
        for overlap in overlapping_strings:
            match = re.search(overlap, input_str)
            if match:
                overlapping_match = (overlap, match.start(), match.end())
                break
        
        # If an overlapping substring is found, mark its position
        if overlapping_match:
            overlap_str, start, end = overlapping_match
            
            # Remove the overlapping substring temporarily
            remaining_str = input_str[:start] + input_str[end:]
            
            # Apply the original logic to the remaining string
            processed_str = remove_subsequences(remaining_str, max_replacements.copy())
            processed_str = processed_str.replace('.', '')

            # Now, process the overlapping substring based on remaining limits
            # We remove the part of the overlap whose limit has not been exhausted
            overlap_limits = max_replacements.copy()

            # Adjust the limits based on what was already removed from the processed string
            for sub in overlap_limits:
                count_removed = len(re.findall(sub, remaining_str))  # Count how many were removed
                overlap_limits[sub] = max(0, overlap_limits[sub] - count_removed)

            # Apply the remaining limits to the overlapping substring
            processed_overlap = remove_subsequences(overlap_str, overlap_limits)
            processed_overlap = processed_overlap.replace('.','')

            # Reconstruct the final processed string
            final_str = processed_str[:start] + processed_overlap + processed_str[start:]
        else:
            # If no overlapping substring, apply the original removal logic
            final_str = removedd_subsequences(input_str, max_replacements.copy())
            final_str = final_str.replace('.', '')

        return final_str

    def process_df(df):
        new_df = df.copy()
        new_df['processed_str'] = new_df['input_str'].apply(process_string)
        
        # Find rows where processed strings do not have length 13
        non_matching_df = new_df[new_df['processed_str'].str.len() != 13]
        
        # Keep only rows where processed strings have length 13
        matching_df = new_df[new_df['processed_str'].str.len() == 13]
        
        # Drop the original 'input_str' column and rename the processed column to 'input_str'
        matching_df = matching_df.drop(columns=['input_str']).rename(columns={'processed_str': 'input_str'})
        
        return matching_df, non_matching_df

    # Process train and validation datasets
    dftrain2, dftrain_non_matching = process_df(dftrain)
    dfvalid2, dfvalid_non_matching = process_df(dfvalid)
    dftest2, dftest_non_matching = process_df(dftest)
    
    return dftrain2, dftrain_non_matching, dfvalid2, dftest2 ,dfvalid_non_matching








def evaluate_with_different_splits(combined_features_train, labels_train, combined_features_val, labels_val, percentages):
    results = {}

    # Define classifiers and parameter grids for hyperparameter tuning
    classifiers = {
          
        'SVC': SVC(),        
    }

    param_grids = {
     
        'SVC': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        },
       
    }

    for pct in percentages:
        # Split the training set according to the percentage
        subset_size = int(len(combined_features_train) * (pct / 100))
        X_train_subset, y_train_subset = combined_features_train[:subset_size], labels_train[:subset_size]

        print(f"Training with {pct}% of the data")
        for clf_name, clf in classifiers.items():
            print(f"\nTraining {clf_name}...")

            # Perform hyperparameter tuning using GridSearchCV
            grid_search = GridSearchCV(clf, param_grids[clf_name], cv=2, scoring='accuracy', n_jobs=-1, verbose=3)
            print("gridsearch complete")
            grid_search.fit(X_train_subset, y_train_subset)
            print("fitting done")

            # Get the best estimator from the grid search
            best_model = grid_search.best_estimator_

            # Evaluate the model on the validation set
            val_predictions = best_model.predict(combined_features_val)
            
            # Save test predictions to a CSV file
            output_df_combined = pd.DataFrame({
                'predicted_label': val_predictions
            })
            
            output_df_combined.to_csv('test_predictions_combined.csv', index=False)
            print(f"Predictions for combined {clf_name} saved to test_predictions.csv")
            
            
            
            
            accuracy = accuracy_score(labels_val, val_predictions)

            print(f"{clf_name} - Best Params: {grid_search.best_params_}, Validation Accuracy: {accuracy:.4f}")

            # Store the result
            if pct not in results:
                results[pct] = {}
            results[pct][clf_name] = {
                'best_params': grid_search.best_params_,
                'accuracy': accuracy
            }
            
    return results


# Function to train and evaluate models with hyperparameter tuning
def evaluate_with_different_splits2(combined_features_train, labels_train, combined_features_val, labels_val, percentages, combined_features_test):
    results = {}

    # Define classifiers directly with specified parameters
    classifiers = {
        'SVC': SVC(C=10, kernel='rbf')
    }

    for pct in percentages:
        # Split the training set according to the percentage
        subset_size = int(len(combined_features_train) * (pct / 100))
        X_train_subset, y_train_subset = combined_features_train[:subset_size], labels_train[:subset_size]

        print(f"Training with {pct}% of the data")
        for clf_name, clf in classifiers.items():
            print(f"\nTraining {clf_name}...")

            # Train the model directly
            clf.fit(X_train_subset, y_train_subset)

            # Evaluate the model on the validation set
            val_predictions = clf.predict(combined_features_val)
            
            # Predict on the test set
            test_predictions = clf.predict(combined_features_test)
            
            # Save test predictions to a CSV file
            with open('pred_combined.txt', 'w') as f:
                for prediction in test_predictions:
                    f.write(f"{prediction}\n")
            print(f"Predictions for combined {clf_name} saved to test_predictions.csv")
            
            # Calculate validation accuracy
            accuracy = accuracy_score(labels_val, val_predictions)

            print(f"{clf_name} - Validation Accuracy: {accuracy:.4f}")

            # Store the result
            if pct not in results:
                results[pct] = {}
            results[pct][clf_name] = {
                'accuracy': accuracy
            }
            
    return results



# main-function

if __name__ == '__main__':



    dftrain = pd.read_csv("datasets/train/train_text_seq.csv")
    dfvalid = pd.read_csv("datasets/valid/valid_text_seq.csv")
    dftest = pd.read_csv("datasets/test/test_text_seq.csv")
    pd.set_option('display.max_colwidth', None)




    print({dftrain.shape})

        #Define subsequences to remove (this is just for reference now)
    subsequences_to_remove = ['15436', '1596', '000', '464', '614', '262', '422', '284']

    dftrain2, dftrain_non_matching, dfvalid2, dfvalid_non_matching, dftest2, dftest_non_matching = process_dataframes(dftrain, dfvalid, dftest,subsequences_to_remove)

    print({dftrain2.shape})

    dftrain_1 = pd.read_csv("datasets/train/train_emoticon.csv")
    dfvalid_1 = pd.read_csv("datasets/valid/valid_emoticon.csv")
    dftest_1 = pd.read_csv("datasets/test/test_emoticon.csv")

            #Load datasets2
    deep_features = np.load("datasets/train/train_feature.npz")
    valid_features = np.load("datasets/valid/valid_feature.npz")
    test_features = np.load("datasets/test/test_feature.npz")


        # Process the datasets
    dftrain_1 = fixDataframe(dftrain_1)
    dfvalid_1 = fixDataframe(dfvalid_1)
    dftest_1 = fixDataframe(dftest_1)

        # Keep only the necessary columns
    dfvalid_1 = dfvalid_1[[
                'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'label']]
    dftrain_1 = dftrain_1[[
                'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'label']]
            
        # Unique emoji sets for encoding
    unique_emojis_f1_to_f8 = sorted(dftrain_1.iloc[:, :8].stack().unique())
    unique_emojis_f9_to_f13 = sorted(dftrain_1.iloc[:, 8:13].stack().unique())

        # Create mappings for encoding
    emoji_to_index_f1_to_f8 = {emoji: idx for idx, emoji in enumerate(unique_emojis_f1_to_f8)}
    emoji_to_index_f9_to_f13 = {emoji: idx for idx, emoji in enumerate(unique_emojis_f9_to_f13)}

        # Encode training, validation, and test sets
    X_train_encoded_1 = encode_dataset(dftrain_1, emoji_to_index_f1_to_f8, emoji_to_index_f9_to_f13)
    X_valid_encoded_1 = encode_dataset(dfvalid_1, emoji_to_index_f1_to_f8, emoji_to_index_f9_to_f13)
    X_test_encoded_1 = encode_dataset(dftest_1, emoji_to_index_f1_to_f8, emoji_to_index_f9_to_f13)

        # Target labels
    y_train_1 = dftrain_1['label']
    y_valid_1 = dfvalid_1['label']

        # Define the hyperparameters for logistic regression
    param_grid_1 = {
                'penalty': ['l1', 'l2'],
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear']
    }



    log_reg_1 = LogisticRegression(penalty='l1', C=100, solver='liblinear', max_iter=1000)

    # Train the model
    log_reg_1.fit(X_train_encoded_1, y_train_1)

    # Validate the model on the validation set
    y_valid_pred_1 = log_reg_1.predict(X_valid_encoded_1)
    valid_accuracy_1 = accuracy_score(y_valid_1, y_valid_pred_1)
    print(f"Validation Accuracy: {valid_accuracy_1 * 100:.2f}%")

    # Make predictions on the test set
    y_test_pred_1 = log_reg_1.predict(X_test_encoded_1)

    # Save the predictions to a CSV file
    with open('pred_emoticon.txt', 'w') as f:
        for prediction in y_test_pred_1:
            f.write(f"{prediction}\n")
            
    print("Test predictions for dataset1 saved to 'test_predictions.txt'")


        # Model for dataset2

    X_deep = deep_features['features']
    y_deep = deep_features['label']
    X_deep_flat = X_deep.reshape(X_deep.shape[0], -1)

    X_valid_2 = valid_features['features']
    y_valid_2 = valid_features['label']
    X_valid_flat = X_valid_2.reshape(X_valid_2.shape[0], -1)

    X_test_2 = test_features['features']
    X_test_flat = X_test_2.reshape(X_test_2.shape[0], -1)

    X_deep_flat, y_deep = shuffle(X_deep_flat, y_deep, random_state=42)

        

    rf_clf_2 = RandomForestClassifier(max_depth=20, min_samples_split=2, random_state=42)

    # Train the model
    rf_clf_2.fit(X_deep_flat, y_deep)

    # Validate the model on the validation set
    y_valid_pred_2 = rf_clf_2.predict(X_valid_flat)
    valid_accuracy_2 = accuracy_score(y_valid_2, y_valid_pred_2)
    print(f"Validation Accuracy: {valid_accuracy_2 * 100:.2f}%")

    # Make predictions on the test set
    y_test_pred_2 = rf_clf_2.predict(X_test_flat)

    # Save the predictions to a text file
    with open('pred_deepfeat.txt', 'w') as f:
        for prediction in y_test_pred_2:
            f.write(f"{prediction}\n")

    print("Test predictions for dataset2 saved to 'test_predictions.txt'.")




    X_train = dftrain2['input_str']
    y_train = dftrain2['label']
    X_train_digits = pd.DataFrame([list(map(int, list(x))) for x in X_train])

    X_valid = dfvalid2['input_str']
    y_valid = dfvalid2['label']
    X_valid_digits = pd.DataFrame([list(map(int, list(x))) for x in X_valid])

    X_test = dftest2['input_str']
    X_test_digits = pd.DataFrame([list(map(int, list(x))) for x in X_test])

                
        # XGBoost model
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic', 
        eval_metric='logloss', 
        use_label_encoder=False,
        learning_rate=0.1, 
        max_depth=6, 
        n_estimators=1400,
        random_state=42
    )

    # Train the model
    xgb_clf.fit(X_train_digits, y_train)

    # Validation accuracy
    y_valid_pred = xgb_clf.predict(X_valid_digits)
    valid_accuracy = accuracy_score(y_valid, y_valid_pred)
    print(f"Validation Accuracy: {valid_accuracy * 100:.2f}%")

    # Test predictions
    test_predictions = xgb_clf.predict(X_test_digits)

    # Save test predictions to a text file
    with open('pred_textseq.txt', 'w') as f:
        for prediction in test_predictions:
            f.write(f"{prediction}\n")

    print("Predictions for dataset3 saved to test_predictions.txt")

    print({dftrain2.shape})



    #task2


    dataset1_train = pd.read_csv("datasets/train/train_emoticon.csv")

    dataset1_train = fixDataframe(dataset1_train)
    dataset1_train = dataset1_train[['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'label']]

    unique_emojis_f1_to_f8 = sorted(dataset1_train.iloc[:, :8].stack().unique())
    unique_emojis_f9_to_f13 = sorted(dataset1_train.iloc[:, 8:13].stack().unique())

    # Create mappings of emojis to indices for one-hot encoding
    emoji_to_index_f1_to_f8 = {emoji: idx for idx, emoji in enumerate(unique_emojis_f1_to_f8)}
    emoji_to_index_f9_to_f13 = {emoji: idx for idx, emoji in enumerate(unique_emojis_f9_to_f13)}

    # Encode dataset 1
    dataset1_train_encoded = encode_dataset(dataset1_train, emoji_to_index_f1_to_f8, emoji_to_index_f9_to_f13)

    # --------------- Load Dataset 2 ---------------
    # Load Dataset 2 (NPZ) for training
    dataset2_train = np.load("datasets/train/train_feature.npz")
    features_2_train = dataset2_train['features']

    # Flatten the 13x768 matrices in Dataset 2 for training
    features_2_flattened_train = features_2_train.reshape(features_2_train.shape[0], -1)

    # --------------- Load Dataset 3 ---------------
    # Load Dataset 3 (CSV) for training and validation
    dataset3_train = pd.read_csv("datasets/train/train_text_seq.csv")
    dataset3_val = pd.read_csv("datasets/valid/valid_text_seq.csv")

    dataset3_test = pd.read_csv("datasets/test/test_text_seq.csv")




    # Define subsequences to remove
    subsequences_to_remove = ['15436', '1596', '000', '464', '614', '262', '422', '284']

    dftrain2, dftrain_non_matching, dfvalid2, dftest2, dfvalid_non_matching  = process_dataframes2(dataset3_train, dataset3_val, dataset3_test,subsequences_to_remove)
    X_train_full_for_dataset3 = dftrain2['input_str']
    y_train_full_for_dataset3 = dftrain2['label']


    X_train_digits = pd.DataFrame([list(map(int, list(x))) for x in X_train_full_for_dataset3])

    X_valid_for_dataset3 = dfvalid2['input_str']
    y_valid_for_dataset3 = dfvalid2['label']

    X_test_for_dataset3 = dftest2['input_str']

    X_valid_digits = pd.DataFrame([list(map(int, list(x))) for x in X_valid_for_dataset3])
    X_test_digits = pd.DataFrame([list(map(int, list(x))) for x in X_test_for_dataset3])

    # Standardize all features
    scaler_2 = StandardScaler()
    features_2_train_scaled = scaler_2.fit_transform(features_2_flattened_train)

    scaler_3 = StandardScaler()
    features_3_train_scaled = scaler_3.fit_transform(X_train_digits)

    # Concatenate Dataset 1 and Dataset 2 for training


    features_2_train_scaled = features_2_train_scaled[:7079, :]
    dataset1_train_encoded = X_train_encoded_1[:7079, :] 
    features_3_train_scaled = features_3_train_scaled[:7079,:]


    combined_features_train = np.hstack((features_2_train_scaled, features_3_train_scaled, dataset1_train_encoded))

    # Prepare training labels
    labels_train = dataset1_train['label'].values

    # --------------- Load Validation Datasets ---------------
    # Load Dataset 1 (CSV) for validation
    dataset1_val = pd.read_csv("datasets/valid/valid_emoticon.csv")
    dataset1_test = pd.read_csv("datasets/test/test_emoticon.csv")

    dataset1_val = fixDataframe(dataset1_val)
    dataset1_val = dataset1_val[['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'label']]
    dataset1_val_encoded = encode_dataset(dataset1_val, emoji_to_index_f1_to_f8, emoji_to_index_f9_to_f13)


    dataset1_test = fixDataframe(dataset1_test)
    dataset1_test = dataset1_test[['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13']]
    dataset1_test_encoded = encode_dataset(dataset1_test, emoji_to_index_f1_to_f8, emoji_to_index_f9_to_f13)


    # Load Dataset 2 (NPZ) for validation
    dataset2_val = np.load("datasets/valid/valid_feature.npz")
    dataset2_test = np.load("datasets/test/test_feature.npz")

    features_2_val = dataset2_val['features']
    features_2_test = dataset2_test['features']

    # Flatten the 13x768 matrices in Dataset 2 for validation
    features_2_flattened_val = features_2_val.reshape(features_2_val.shape[0], -1)
    features_2_flattened_test = features_2_test.reshape(features_2_test.shape[0], -1)

    # Standardize all validation features
    features_2_val_scaled = scaler_2.transform(features_2_flattened_val)
    features_3_val_scaled = scaler_3.transform(X_valid_digits)

    features_2_test_scaled = scaler_2.transform(features_2_flattened_test)
    features_3_test_scaled = scaler_3.transform(X_test_digits)




    # Concatenate Dataset 1 and Dataset 2 for validation
    combined_features_val = np.hstack((features_2_val_scaled, features_3_val_scaled, dataset1_val_encoded))


    combined_features_test = np.hstack((features_2_test_scaled, features_3_test_scaled, dataset1_test_encoded))

    # Prepare validation labels
    labels_val = dataset1_val['label'].values



    # --------------- Evaluate Models ---------------
    percentages = [80]  # Percentages of training data to use
    results = evaluate_with_different_splits2(combined_features_train, labels_train, combined_features_val, labels_val, percentages,combined_features_test)

    # --------------- Plot the Results ---------------
    # Extract the results for each classifier
    # for clf_name in ['RandomForest', 'LogisticRegression', 'SVC', 'XGBoost']:
    #     x = percentages
    #     y = [results[pct][clf_name]['accuracy'] for pct in percentages]

    #     # Plot the percentages vs accuracy for each classifier
    #     plt.plot(x, y, marker='o', linestyle='-', label=f'{clf_name} Accuracy')

    # plt.title('Validation Accuracy vs. Training Data Percentage')
    # plt.xlabel('Percentage of Training Data Used')
    # plt.ylabel('Validation Accuracy')
    # plt.xticks(percentages)
    # plt.grid(True)
    # plt.legend()
    # plt.show()

