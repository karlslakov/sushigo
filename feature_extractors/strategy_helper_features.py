import numpy as np
from constants import nigiri_scores
import gch

class strategy_helper_features:
    def extract(self, player, game):
        cplayer = player

        out = []

        for _ in range(game.players):
            cselected = np.array(game.curr_round_hands[cplayer])
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
            out.append(int(wassabi_active))
            out.append(int(needs_tempura))
            out.append(int(needs_sashimi))
            cplayer = gch.get_next_player(cplayer, game)
        
        return np.array(out)
    
    def output_size(self, players):
        return 3 * players