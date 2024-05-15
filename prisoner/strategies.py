'''This code contains the next move strategies for the prisoner dillema game
by Jan Stasinski'''
import numpy as np

def mr_nice_move(round, game_hist, pidx):
    return 1 # Always cooperate

def mr_mean_move(round, game_hist, pidx):
    return 0 # Always defect

def tit4tat_move(round, game_hist, pidx):
    return game_hist[1-pidx, round-1] # Copy the last move of the opponent

def tat4tit_move(round, game_hist, pidx):
    return abs(game_hist[1-pidx, round-1]-1)

def tit4_2tats_move(round, game_hist, pidx):
    if round == 1:
        return 1
    if game_hist[1-pidx, round-1] == 0 and game_hist[1-pidx, round-2] == 0:
        return 0
    elif game_hist[1-pidx, round-1] == 1 and game_hist[1-pidx, round-2] == 1:
        return 1
    else:
        return 1

def tat4_2tits_move(round, game_hist, pidx):
    if round == 1:
        return 0
    if game_hist[1-pidx, round-1] == 0 and game_hist[1-pidx, round-2] == 0:
        return 0
    elif game_hist[1-pidx, round-1] == 1 and game_hist[1-pidx, round-2] == 1:
        return 1
    else:
        return 0

def fool_me_once_move(round, game_hist, pidx):
    if all(game_hist[1-pidx, :]) == 0:
        return 1
    else:
        return 0

def fool_me_Xtimes_move(round, game_hist, pidx, X):
    if round < X:
        return np.random.randint(0,2)
    if all(game_hist[1-pidx, round-X: round] == 0):
        return 0
    else:
        return 1

def you_2_nice_move(round, game_hist, pidx):
    if round < 2:
        return 0
    elif all(game_hist[1-pidx, round-3: round] == 1):
        return 0
    else:
        return 0

def punish_Xtimes_move(round, game_hist, pidx, X):
    if round < X:
        return 1
    elif game_hist[1-pidx, round-1] == 0:
        return 0
    elif all(game_hist[pidx, round-X:round]) == 0:
        return 1
    else:
        return 1

def mr_rnd_move(round, game_hist, pidx):
    return np.random.randint(0,2)

def rnd_but_learning_move(round, game_hist, pidx, X):
    if round < 2:
        return np.random.randint(0,2)
    elif all(game_hist[1-pidx, round-3: round] == 1):
        return 0
    else:
        return np.random.randint(0,2)

