import numpy as np
from constants import maki_counts, nigiri_scores, dumping_scores
from feature_extractors.extractor_helpers import onehot_len, to_int
import torch

output_size = onehot_len # int(onehot_len + onehot_len * (onehot_len + 1) / 2)

def get_invalid_outputs(chand, has_chopsticks):
    invalids = np.zeros(output_size, dtype='int32')
    invalids[chand == 0] = 1
    return invalids

def remove_invalid_outputs(output, chand, has_chopsticks):
    output[get_invalid_outputs(chand, has_chopsticks) == 1] = float('-inf')
    return output

def parse_output(output, chand, selected, has_chopsticks):
    invalids = get_invalid_outputs(chand, has_chopsticks)
    with torch.no_grad():
        probs = torch.exp(output).detach().numpy()
        # probs[invalids == 1] = 0
        # probs[invalids == 0] += 0.01
        # probs /= probs.sum()
        avg_invalid_
        command = np.random.choice(range(len(probs)), p=probs)
        orig = command
        is_invalid = False
        if invalids[command] == 1:
            is_invalid = True
            rand_select = np.random.rand(onehot_len)
            rand_select[invalids == 1] = float('-inf')
            command = np.argmax(rand_select)
        return command, is_invalid, orig

def get_action(index):
    return index

def get_shz(players):
    return 12 - players

def get_clockwise_player(p, nump):
    return (p + 1) % nump
def get_cclockwise_player(p, nump):
    return (p - 1) % nump
def get_next_player(p, game):
    if game.round == 1:
        return get_cclockwise_player(p, game.players)
    else:
        return get_clockwise_player(p, game.players)

def calculate_intermediate_score(selected):
    wasabi = 0
    score = 0
    sashimi = 0
    tempura = 0
    dumplings = 0
    for c in selected:
        if c == 'w':
            wasabi += 1
        elif c in nigiri_scores:
            if wasabi > 0:
                score += nigiri_scores[c] * 3
                wasabi -= 1
            else:
                score += nigiri_scores[c]
        elif c == 's':
            sashimi += 1
            if sashimi == 3:
                sashimi = 0
                score += 10
        elif c == 't':
            tempura += 1
            if tempura == 2:
                tempura = 0
                score += 5
        elif c == 'd':
            dumplings += 1
        else:
            continue
    if dumplings > 5:
        dumplings = 5
    score += dumping_scores[dumplings]
    return score

def calculate_final_score(selected, final_round):
    mk_counts = np.array([sum(maki_counts[x] for x in s if x in maki_counts) for s in selected])
    round_scores = np.array([calculate_intermediate_score(s) for s in selected])
    max_mk = np.max(mk_counts)
    mk_winners = mk_counts == max_mk
    num_winners = np.count_nonzero(mk_winners)
    bonus_scores = np.zeros(len(selected))

    bonus_scores[mk_winners] += 6 // num_winners

    if num_winners == 1:
        mk_counts[mk_winners] = -1
        smax_mk = np.max(mk_counts)
        mk_runnersup = mk_counts == smax_mk
        num_runnersup = np.count_nonzero(mk_runnersup)

        bonus_scores[mk_runnersup] += 3 // num_runnersup
    
    if final_round:
        p_counts = np.array([sum(1 for x in s if x == 'p') for s in selected])
        max_p = np.max(p_counts)
        p_winners = p_counts == max_p
        num_winners = np.count_nonzero(p_winners)

        bonus_scores[p_winners] += 6 // num_winners

        if len(selected) > 2:
            min_p = np.min(p_counts)
            p_losers = p_counts == min_p
            num_losers = np.count_nonzero(p_losers)

            bonus_scores[p_losers] -= 6 // num_losers 
    return round_scores + bonus_scores

def get_reward(true_scores, temp_scores, game_over, player):
    reward = 0
    if game_over:
        # technically "noisy" cause of ties but im sure big boy can handle it
        argsorted = np.argsort(true_scores)
        places = argsorted.tolist()
        place = places.index(player)
        reward = true_scores[player] + 35 * place
    else:
        # implement round based punishment for losers?
        reward = temp_scores[player]
    return reward
