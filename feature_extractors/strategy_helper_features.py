import numpy as np
from constants import nigiri_scores
import gch

class strategy_helper_features:
    def extract(self, player, game):
        cplayer = player

        out = []

        for _ in range(game.players):
            cselected = np.array(game.selection_ordered[cplayer])
            ntemp = np.count_nonzero(cselected == 't')
            nsashimi = np.count_nonzero(cselected == 's')
            needs_tempura = ntemp % 2 == 1
            needs_sashimi = nsashimi % 3 == 2
            # in fact, these are almost exaclty the same as the sashimi/tempura bucket in player_selected_features

            wasabi = 0
            for c in cselected:
                if c == 'w':
                    wasabi += 1
                elif c in nigiri_scores and wasabi > 0:
                    wasabi -= 1
            wasabi_active = wasabi > 0
            out.append(int(wasabi_active))
            out.append(int(needs_tempura))
            out.append(int(needs_sashimi))
            cplayer = gch.get_next_player(cplayer, game)
        
        return np.array(out)
    
    def output_size(self, players):
        return 3 * players