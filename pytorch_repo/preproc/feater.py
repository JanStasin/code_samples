import numpy
from data_prep import *

# SELECT A SELECTION METHOD:
#sel_method = 'KBest'
sel_method = 'RFE'

# RUN THE SELECTION
fs = featSelect('nba_data/stats_full_merged.csv', 'positions', cats=None,
            exclude_feats=['slug'], normalize=True,
            describeX=False, regress=False, method=sel_method, met_comp=True, k=15, verbose=True)
print(fs)
if sel_method == 'KBest':
    sel_fts = fs[1]['mutual_info_classif'][0]
    print(f'Selected_features {sel_fts }')
elif sel_method == 'RFE':
    sel_fts = fs[1]
    print(f'Selected_features {sel_fts }')

#get only the selected high scoring features:
cm = pd.DataFrame(fs[0])
cm2 = cm.loc[list(sel_fts),list(sel_fts)]

saveCorrMatrix(cm2, filename='corr_matrix_select_full.png', save_plot=True)

def identify_pairs(corr_matrix, threshold, verbose=False):
    # Ensure corr_matrix is a NumPy array
    corr_matrix_arr = np.array(corr_matrix)
    
    # Disregard the diagonal values
    np.fill_diagonal(corr_matrix_arr, 0.)
    
    # Initialize an empty list to store pairs of features and their correlation coefficients
    pairs_list = []
    
    # Iterate through the upper triangle of the correlation matrix
    for i in range(corr_matrix_arr.shape[0]):
        for j in range(i+1, corr_matrix_arr.shape[1]):
            # Check if the absolute value of the correlation coefficient is above the threshold
            if abs(corr_matrix_arr[i, j]) > threshold:
                # Append the pair of features and their correlation coefficient to the list
                pairs_list.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix_arr[i, j]))
    
    # Print out the pairs of features with their correlation coefficients
    if verbose:
        for pair in pairs_list:
            print(f"Pair: {pair[0]} and {pair[1]}, Correlation: {pair[2]}")
    
    return pairs_list

#cs = identify_pairs(cm2, 0.75, verbose=False)
#print(f'Keeping {cs}')

#print(f'Suggestion to discard: {cs}' )



