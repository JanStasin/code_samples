'''prisoner dilemma functions by Jan Stasinski'''
import numpy as np
from strategies import *
import matplotlib.pyplot as plt

COOPERATE = 1
DEFECT = 0

strategy_functions = {
    'mr_nice': mr_nice_move,
    'mr_mean': mr_mean_move,
    'tit4tat': tit4tat_move,
    'tat4tit': tat4tit_move,
    'tit4_2tats': tit4_2tats_move,
    'tat4_2tits': tat4_2tits_move,
    'fool_me_once': fool_me_once_move,
    'fool_me_Xtimes': fool_me_Xtimes_move,
    'you_2_nice': you_2_nice_move,
    'punish_Xtimes': punish_Xtimes_move,
    'mr_rnd': mr_rnd_move,
    'rnd_but_learning': rnd_but_learning_move }

def single_round(dA, dB):
    if dA == 1 and dB == 1:
        out = 'CC'
        return 3, 3, out
    elif dA == 1 and dB == 0:
        out = 'CD'
        return 0, 5, out
    elif dA == 0 and dB == 1:
        out = 'DC'
        return 5, 0, out
    elif dA == 0 and dB == 0:
        out = 'DD'
        return 1, 1, out
    else:
        raise ValueError('Invalid input for single_round: dA and dB must be 0 or 1')
        
class Player:    
    types = ['mr_rnd', 'rnd_but_learning', 'mr_nice', 'tit4tat', 'tit4_2tats',
              'fool_me_Xtimes', 'fool_me_once', 'mr_mean', 'tat4tit', 'tat4_2tits', 
              'you_2_nice', 'punish_Xtimes']
    
    def __init__(self, type_idx, history=None):
        self.type = self.types[type_idx]
        #print(f'This player uses {self.type} strategy')
        if history is None:
            history = []
        self.history = history

    def __str__(self):
        return f'{self.type}'
    
    def first_move(self):
        if self.type in ['mr_nice', 'tit4tat', 'tit4_2tats', 'fool_me_once', 'fool_me_Xtimes', ]:
            return 1
        elif self.type in ['mr_mean', 'tat4tit', 'tat4_2tits', 'you_2_nice', 'punish_Xtimes']:
            return 0
        elif self.type in ['mr_rnd', 'rnd_but_learning']:
            return np.random.randint(0,2)
        
def insert_noise(var, noise_level=0.1):
    roll = np.random.rand(1)
    if roll < noise_level:
        return 1 - var
    else:
        return var

def next_move(round, strategy, pidx, game_hist, X):
    if strategy in strategy_functions:
        # Check if the strategy requires an 'X' argument
        if strategy in ['fool_me_Xtimes', 'rnd_but_learning', 'punish_Xtimes']:
            return strategy_functions[strategy](round, game_hist, pidx, X)
        else:
            return strategy_functions[strategy](round, game_hist, pidx)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
        
def game(pA, pB, N_rounds, noise=True):
    outcomes_dict = { 'CC': 0, 'CD': 1, 'DC': 2, 'DD': 3}
    game_hist = np.empty([5, N_rounds])
    #print(f'history shape: {game_hist.shape}')
    for round in range(N_rounds):
        if round == 0:
            game_hist[0,round] = pA.first_move()
            game_hist[1,round] = pB.first_move()
            pA_score, pB_score, outcome = single_round(game_hist[0,round], game_hist[1,round])
            game_hist[2,round] = pA_score
            game_hist[3,round] = pB_score
            game_hist[4,round] = outcomes_dict[outcome]
        else:

            game_hist[0,round] = next_move(round, pA.type, 0, game_hist, X=5)
            game_hist[1,round] = next_move(round, pB.type, 1, game_hist, X=5)

            if noise:
                print(insert_noise(game_hist[0,round]), game_hist[0,round])
                pA_score, pB_score, outcome = single_round(insert_noise(game_hist[0,round]), insert_noise(game_hist[1,round]))
                
            else:
                pA_score, pB_score, outcome = single_round(game_hist[0,round], game_hist[1,round])
            # print(game_hist[2,round-1], pA_score , outcome)
            # print(game_hist[3,round-1] , pB_score, outcome)
            game_hist[2,round] = game_hist[2,round-1] + pA_score
            game_hist[3,round] = game_hist[3,round-1] + pB_score
            game_hist[4,round] = outcomes_dict[outcome]

        if round % 40 == 0:
            print(f'Round {round+1}: Player A chose {game_hist[0,round]}, Player B chose {game_hist[1,round]}. Outcome: {outcome}. Scores: Player A: {game_hist[2,round]}, Player B: {game_hist[3,round]}')
    return game_hist

# if __name__ == "__main__":
#     N = 20
#     p1 = Player(np.random.randint(0,12))
#     p2 = Player(np.random.randint(0,12)) 
#     print(f'{p1.type} VS {p2.type} playing {N} rounds')
#     res = game(p1, p2, N, noise=False)
#     print(res[:, -50:])
#     print(f'Player1 {p1.type} scored: {res[2,-1]}, Player2 {p2.type} scored: {res[3,-1]}')


def make_heatmap(hm_array, types, save_path='prisoner/hm.jpg'):
    fig, ax = plt.subplots(figsize=(20,14))
    # Display the heatmap
    i = ax.imshow(hm_array)
    # Set the tick labels for the x-axis
    ax.set_xticks(np.arange(len(types)))
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.set_xticklabels(types, rotation=90) # Rotate labels 
    # Mirror the tick labels for the y-axis
    ax.set_yticks(np.arange(len(types)))
    ax.set_yticklabels(types)
    # Add a colorbar
    cax = fig.add_axes([ax.get_position().x1+0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    fig.colorbar(i, cax=cax)

    # Add text inside each square
    for i in range(hm_array.shape[0]):
        for j in range(hm_array.shape[1]):
            text = ax.text(j, i, int(hm_array[i, j]),
                           ha="center", va="center", color="w")

    # Save and show the figure
    fig.savefig(save_path)
    
def make_barplot(hm_array, types, save_path='prisoner/bp.jpg'):
    fig, ax = plt.subplots(figsize=(16,10))
    sum_data=np.sum(hm_array, axis=0)
    my_cmap = plt.get_cmap('viridis') # Example colormap, replace 'viridis' with your colormap name
    
    # Sort the data get the indices
    sorted_indices = np.argsort(sum_data)
    sorted_data = sum_data[sorted_indices]
    sorted_types = np.array(types)[sorted_indices]
    # Normalize your data to the range [0, 1] for the cm
    norm_data = (sorted_data - sorted_data.min()) / (sorted_data.max() - sorted_data.min())
    # Map the normalized data to colors
    colors = my_cmap(norm_data)
    ax.barh(np.arange(len(sorted_types)), sorted_data, color=colors) 
    ax.set_yticks(np.arange(len(types)))
    ax.set_yticklabels(sorted_types)
    ax.set_xlabel('Total points')
    fig.savefig(save_path)


