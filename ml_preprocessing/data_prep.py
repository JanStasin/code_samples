import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2, f_classif, r_regression, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector

from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler

def featSelect(file_name, target, cats , exclude_feats, normalize=True,
                 describeX=False, regress=False, method=None, met_comp=True, k=8, verbose=False):

    """
    Performs feature selection on a dataset using various methods.

    Parameters:
    - file_name (str): Name of the file containing the dataset.
    - target (str): Column name of the target variable in the dataset.
    - cats (list): List of categorical variables in the dataset.
    - exclude_feats (list): Features to exclude from the dataset.
    - normalize (bool, default True): Flag to indicate whether to normalize the dataset features.
    - describeX (bool, default False): Flag to indicate whether to generate descriptive statistics for each feature.
    - regress (bool, default False): Flag to indicate whether to include regression analysis in the feature selection process.
    - method (str, optional): Feature selection method to apply ('VF', 'KBest', 'RFE', 'boruta'). Default is None.
    - met_comp (bool, default True): Flag to indicate whether to compare methods during feature selection.
    - k (int, default 8): Number of top features to select when using the K-Best method.
    - verbose (bool, default False): Flag to indicate whether to display detailed output during execution.

    Returns:
    - Depending on the method, returns the correlation matrix, selected features, unselected features, variance scores, or a dictionary containing the selected features and their corresponding scores.
    """
    X, Y = process_data(file_name, target, cats, norm=False, feats2drop=exclude_feats)
    cols = X.columns
    
    if normalize:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        X = pd.DataFrame(X, columns=cols)
        
    if describeX:
        stats = X.describe()
        for feature in cols:
            print(f"Feature: {feature}")
            stats = X[feature].describe()
            print(f"Variance: {X[feature].var()}")
            print(f"Median: {stats['50%']}, Mean: {stats['mean']}, Max: {stats['max']}")

    # Calculate Pearson R correlation matrix
    corr_matrix = X.corr()
    if regress:
        # Include correlation with the target variable Y
        corr_matrix[target] = X.corrwith(Y)

    # Get correlation matrix as a heatmap using matplotlib

    saveCorrMatrix(corr_matrix, filename='corr_matrix_full.png', save_plot=True)

    if method == 'VF':
        sel_features, unsel_features, vs = varThres(Xnorm, 0.05, verbose)
        return corr_matrix, sel_features, unsel_features, vs

    if method == 'KBest':
        features_dict = featUniMethods(X, Y, k=k, regress=regress, met_comp=met_comp, verbose=verbose)
        #if features_dict['f_classif']: 
        return corr_matrix, features_dict

    if method == 'RFE':
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25)
        features_dict = runRFE(X_train, Y_train, X_test, Y_test, kN=k, make_plot=True)
        return corr_matrix, features_dict
    if method == 'boruta':
        pass

def varThres(Xnorm, thres_value=0.05, verbose=False, make_plot=True):
    '''
    Perform variance thresholding on a normalized df of features and return the selected and unselected features.
    
    Parameters:
    - Xnorm (pd.DataFrame): The input dataset.
    - thres_value (float): The threshold value for variance.
    - verbose (bool): If True, print detailed information about the selected features.
    
    Returns:
    - selected_features (pd.Index): The names of the selected features.
    - unselected_features (pd.Index): The names of the unselected features.
    - variances (pd.Series): The variance of each selected feature.
    '''
    if not isinstance(Xnorm, pd.DataFrame):
        raise ValueError("Xnorm must be a pandas DataFrame.")
    
    VT = VarianceThreshold(threshold=thres_value)
    VT.fit_transform(Xnorm)

    selection_info = VT.get_support()
    selected_features = Xnorm.columns[selection_info]
    unselected_features = Xnorm.columns[~selection_info]
    
    variances = Xnorm[selected_features].var()
    
    if verbose:
        print("Selected features:", selected_features)
        for feature in selected_features:
            print(f"Feature: {feature}, Variance: {variances[feature]}")
        print("Unselected features:", unselected_features)

    if make_plot:
        fig, ax = plt.subplots(figsize=(12,10))

        x = Xnorm.columns
        y = np.round(Xnorm.var(axis=0), 4)

        ax.bar(x, y, width=0.2)
        ax.set_xlabel('Features')
        ax.set_ylabel('Variance')
        ax.set_ylim(0, max(y))


        for index, value in enumerate(y):
            plt.text(x=index, y=value+0.0001, s=str(round(value, 3)), ha='center')
            
        fig.autofmt_xdate()
        plt.tight_layout()
        fig.show()
        fig.savefig('norm_feat_var_plot.png')
    
    return selected_features, unselected_features, variances

def featUniMethods(X, Y, k=5, regress=False, met_comp=False, verbose=False):
    '''
    Applies univariate feature selection methods to a dataset.

    Parameters:
    - X (DataFrame): Input features DataFrame.
    - Y (Series): Target variable Series.
    - k (int, default 5): Number of top features to select.
    - regress (bool, default False): Flag to indicate whether to use regression-based methods.
    - met_comp (bool, default False): Flag to indicate whether to compare multiple methods.
    - verbose (bool, default False): Flag to indicate whether to display detailed output.

    Returns:
    - features_dict (dict): Dictionary containing the selected features and their corresponding scores and p-values.
    '''

    features_dict = {}
    # Define the method to be used for feature selection
    if regress:
        methods = [f_regression, r_regression, mutual_info_regression]
        m_names = ['f_regression', 'r_regression', 'mutual_info_regression']
    else:
        methods = [chi2, f_classif, mutual_info_classif]
        m_names = ['chi2', 'f_classif', 'mutual_info_classif']

    if met_comp:
        for midx, met in enumerate(methods):
            SKB = SelectKBest(met, k=k)
            SKB.fit_transform(X, Y)
            # Get the names of the selected features
            selected_features = X.columns[SKB.get_support()]
            # Get the classification scores and p-values for the selected features
            selected_scores = SKB.scores_[SKB.get_support()]
            sorted_indices = np.argsort(selected_scores)[::-1]
            if m_names[midx] not in ['r_regression', 'mutual_info_regression', 'mutual_info_classif']: selected_pvalues = SKB.pvalues_[SKB.get_support()]
                    # Prepare the data to be stored in the dictionary
            if regress and m_names[midx] == 'r_regression':
                features_dict[m_names[midx]] = [selected_features[sorted_indices], selected_scores[sorted_indices]]
            else:
                features_dict[m_names[midx]] = [selected_features[sorted_indices], selected_scores[sorted_indices], selected_pvalues[sorted_indices]]
            print(f'{m_names[midx]} - Selected features -> {features_dict[m_names[midx]][0]}')
    else:
        SKB = SelectKBest(methods[2], k=k)
        SKB.fit_transform(X, Y)
        # Get the names of the selected features
        selected_features = X.columns[SKB.get_support()]
        # Get the classification scores and p-values for the selected features
        selected_scores = SKB.scores_[SKB.get_support()]
        sorted_indices = np.argsort(selected_scores)[::-1]
        features_dict[m_names[2]] = [selected_features[sorted_indices], selected_scores[sorted_indices]]
        print(f'{m_names[2]} - Selected features -> {features_dict[m_names[2]][0]}')

        # Print the sorted features, their scores, and p-values if verbose is True
        if verbose and met_comp:
            print(f'{m_names[midx]}')
            for idx in sorted_indices:
                print(f'Feature: {selected_features[idx]}, Score: {selected_scores[idx]}')
        else:
            print(m_names[2])
            for idx in sorted_indices:
                print(f'Feature: {selected_features[idx]}, Score: {selected_scores[idx]}')
    
    return features_dict


def runRFE(X_train, Y_train, X_test, Y_test, kN=None, make_plot=True):
    '''
    Runs Recursive Feature Elimination (RFE) on a dataset and plots the F1-score against the number of features selected.

    Parameters:
    - X_train (DataFrame): Training set features.
    - Y_train (Series): Training set labels.
    - X_test (DataFrame): Test set features.
    - Y_test (Series): Test set labels.
    - kN (int, optional): Number of features to consider. Defaults to the total number of features in X_train.
    - make_plot (bool, default True): Flag to indicate whether to plot the F1-scores.

    Returns:
    - features_dict (dict): Dictionary containing the selected features, optimal number of features, and the maximum F1-score achieved.
    '''
    rfe_scores = []
    if not kN: kN = len(X_train.columns)
    
    for k in range(1, kN):
        gbc = GradientBoostingClassifier(max_depth=15, random_state=42)
        RFE_selector = RFE(estimator=gbc, n_features_to_select=k, step=1)
        RFE_selector.fit(X_train, Y_train)
        
        sel_X_train = RFE_selector.transform(X_train)
        sel_X_test = RFE_selector.transform(X_test)
        
        gbc.fit(sel_X_train, Y_train)
        RFE_predicts = gbc.predict(sel_X_test)
        
        RFE_score = round(f1_score(Y_test, RFE_predicts, average='weighted'), 5)
        rfe_scores.append(RFE_score)

    if make_plot:
        fig, ax = plt.subplots(figsize=(16,4))
        x = np.arange(1, kN)
        y = rfe_scores

        ax.bar(x, y, width=0.2)
        ax.set_xlabel('Number of features selected using RFE')
        ax.set_ylabel('F1-Score (weighted)')
        ax.set_ylim(0, 1.2)
        ax.set_xticks(np.arange(1, kN))
        ax.set_xticklabels(np.arange(1, kN), fontsize=12)
        for i, v in enumerate(y):
            plt.text(x=i+1, y=v+0.05, s=str(v), ha='center')
        fig.tight_layout()

    selN = rfe_scores.index(max(rfe_scores))
    RFE_selector = RFE(estimator=gbc,n_features_to_select=selN, step=10)
    RFE_selector.fit(X_train, y_train)

    selected_features = X_train.columns[RFE_selector.get_support()]
    features_dict = {}
    features_dict[f'RFE'] = [selected_features, selN, rfe_scores[selN+1]]

    return features_dict

def getAccuracy(y_pred, y_test):
    #print(y_pred == y_test)
    acc = np.sum(y_pred == y_test)/len(y_test)
    return acc

def check_types(data):
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{key}: {type(value)}")
            check_types(value)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            print(f"Index {i}: {type(item)}")
            check_types(item)
    elif isinstance(data, np.ndarray):
        for i in range(data.shape[0]):
            print(f"Index {i}: {type(data[i])}")
            check_types(data[i])
    else:
        print(f"{data}: {type(data)}")
    

def process_data(file_name, Y_col, chosen_cats,  norm=True, feats2drop=None, chosen_feats=None):
    """
    Processes data from a CSV file, applying optional feature selection, normalization, and encoding.

    Parameters:
    - file_name (str): The path to the CSV file containing the data.
    - positions (list): A list of values to filter the data by.
    - target_col (str): The name of the column to be used as the target variable.
    - norm (bool, optional): Whether to normalize the features. Default is True.
    - feats2drop (list, optional): A list of feature names to drop from the dataset.
    - chosen_feats (list, optional): A list of feature names to keep in the dataset.

    Returns:
    - X_scaled (array-like): The processed features, optionally normalized.
    - Y (array-like): The target variable.
    """
   
    df = pd.read_csv(file_name)
    if chosen_cats:
        filtered_df = df[df[Y_col].isin(chosen_cats)]
    else:
        filtered_df = df

    # Categorical encoding
    if filtered_df[Y_col].dtype.name in ['category', 'object']:
        print('encoding categorical data')
        encoder = OrdinalEncoder()
        encoded_positions = encoder.fit_transform(filtered_df[[Y_col]])
        filtered_df.loc[:, Y_col] = encoded_positions

    Y = filtered_df[Y_col]
    Y = np.array([int(y) for y in Y])
    
    # Handle feature selection
    if chosen_feats:
        X = filtered_df[chosen_feats]
    elif feats2drop:
        X = filtered_df.drop(feats2drop, axis=1)
        X = X.drop(Y_col, axis=1) # Exclude the target column from features

    if not chosen_feats and not feats2drop:
        raise ValueError("At least one of 'chosen_feats' or 'feats2drop' must be provided.")
    
    print(f'X features after inital processing: {list(X.columns)}')
    # Normalize features if required
    if norm:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    
    return X_scaled, Y


def saveCorrMatrix(corr_matrix, filename='full_fear_R.png', save_plot=True):
    """
    Saves a plot of the correlation matrix as a PNG file using fig, ax = plt.subplots().

    Parameters:
    - corr_matrix (pd.DataFrame): The correlation matrix to plot.
    - filename (str): The name of the file to save the plot as. Default is 'full_fear_R.png'.
    """
    # Create a figure and axes object
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
    
    # Add a color bar
    fig.colorbar(im, ax=ax)

    # Set x-ticks with labels and rotation
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=90)
    # Set y-ticks with labels
    ax.set_yticks(range(len(corr_matrix.columns)))
    ax.set_yticklabels(corr_matrix.columns)
    # Save the plot as a PNG file
    if save_plot:
        fig.savefig(filename)
    fig.show()