import feature_extractors.extractor_helpers as exh
import gch
import numpy as np

class player_hand_features:
    def extract(self, player, game):
        cplayer = player
        out = []
        for p in range(game.players):
            # if p <= game.in_round_card:
            x = game.curr_round_hands[cplayer]
            out.append(x)
            out.append(x > 0)
            
            # else:
            #    out.append(np.zeros(exh.onehot_len))
            cplayer = gch.get_next_player(cplayer, game)
        return np.array(out)
    
    def output_size(self, players):
        return players * exh.onehot_len * 2