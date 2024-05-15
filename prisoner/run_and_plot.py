
'''prisoner dilemma code to run the simulations and generate the heatmap by Jan Stasinski'''
import numpy as np
from strategies import *
#from prisoner import Player, single_round, game, next_move, make_heatmap, strategy function
from prisoner import *
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

types = list(strategy_functions.keys())
strategy_pairs = list(combinations(types, 2))

##prepare the output array for plotting
hm_array = np.zeros([len(types), len(types)])

if __name__ == "__main__":
    N = 200
    res_dict = {}
    for sp in strategy_pairs:
        print(sp, types.index(sp[0]),types.index(sp[1]))
        p1 = Player(types.index(sp[0]))
        p2 = Player(types.index(sp[1])) 
        print(f'{p1.type} VS {p2.type} playing {N} rounds')
        res = game(p1, p2, N, noise=True)
        print(f'Player1 {p1.type} scored: {res[2,-1]}, Player2 {p2.type} scored: {res[3,-1]}')
        comb_key = sp[0]+'_VS_'+ sp[1]
        if res is not None: # Ensure res is not None before using it
            comb_key = sp[0]+'_VS_'+ sp[1]
            res_dict[comb_key] = res
            hm_array[types.index(sp[0]), types.index(sp[1])] = res[2,-1]
            hm_array[types.index(sp[1]), types.index(sp[0])] = res[3,-1]
        else:
            print(f"No result for {sp[0]} vs {sp[1]}")


make_heatmap(hm_array, types, save_path='prisoner/hm.jpg')
make_barplot(hm_array, types, save_path='prisoner/bp.png')