import numpy as np
from constants import maki_counts, nigiri_scores


def remove_invalid(output, shz, chandsize, has_chopsticks):
    for i in range(len(output)):
        first, chopsticks, second = get_action(i, shz)
        if first >= chandsize or second >= chandsize or chopsticks and not has_chopsticks:
            output[i] = -1
    return output    

def parse_output(output, shz, chandsize, selected):
    output = remove_invalid(output, shz, chandsize, 'c' in selected)
    index = np.argmax(output)
    return get_action(index, shz)

def get_action(index, shz):
    if index < shz:
        return index, False, 0
    first = -1
    while index >= shz:
        index = index - shz
        shz -= 1
        first += 1
    return first, True, first+index+1


def get_clockwise_player(p, nump):
    return (p + 1) % nump
def get_cclockwise_player(p, nump):
    return (p - 1) % nump

def calculate_intermediate_score(selected):
    wasabi = 0
    score = 0
    sashimi = 0
    tempura = 0
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
        else:
            continue
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