import sys
import os
# Add the parent directory of the pytorch directory to the system path
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# # import the module
from preproc.data_prep import process_data
from src.hypertune_func import hyperparameter_tuning

### DEFINE PARAMETER GRID

param_grid = {
        'n_epochs': [100, 500, 1000, 1500],
        'h_nodes': [(12,6),(20, 10), (30, 15),(100,5)],
        'l': [0.001, 0.0005, 0.00001]
    }

if __name__ == "__main__":
    pos = ['POINT GUARD', 'SHOOTING GUARD', 'SMALL FORWARD', 'POWER FORWARD', 'CENTER']
    chosen_feats = ['total_rebound_percentage', 'assist_percentage', 'steal_percentage','block_percentage', 
                    'three_point_attempt_rate', 'offensive_rebounds', 'blocks', 'assists', 'free_throw_attempt_rate','attempted_three_point_field_goals']

    X_scaled, Y = process_data('data/nba_stats_full_merged.csv', 'positions', pos, chosen_feats=chosen_feats)
    print(X_scaled.shape,  Y.shape)

    # Call the hyperparameter tuning function
    best_params, best_score = hyperparameter_tuning(X_scaled, Y, param_grid)
    print(f"Best Parameters: {best_params}")
    print(f"Best Score: {best_score}")