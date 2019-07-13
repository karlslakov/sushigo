import numpy as np
from constants import nigiri_scores

class strategy_helper_features:
    def extract(self, player, game):
        cselected = np.array(game.curr_round_hands[player])
        ntemp = np.count_nonzero(cselected == 't')
        nsashimi = np.count_nonzero(cselected == 's')
        needs_tempura = ntemp % 2 == 1
        needs_sashimi = nsashimi % 3 == 2
        # in fact, these are almost exaclty the same as the sashimi/tempura bucket in player_selected_features

        wassabi = 0
        for c in cselected:
            if c == 'w':
                wassabi += 1
            elif c in nigiri_scores:
                wassabi -= 1
        wassabi_active = wassabi > 0
        return np.array([int(needs_tempura), int(needs_sashimi), int(wassabi_active)])
    
    def output_size(self, game):
        return 3