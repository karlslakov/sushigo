import feature_extractors.extractor_helpers as exh
import gch
import numpy as np

class player_hand_features:
    def extract(self, player, game):
        cplayer = player
        out = []
        for p in range(game.players):
            for c in range(game.shz):
                if c < len(game.curr_round_hands[cplayer]) and p <= game.in_round_card:
                    out.append(exh.to_onehot_embedding(game.curr_round_hands[cplayer][c]))
                else:
                    out.append(exh.nocard_onehot)
            cplayer = gch.get_clockwise_player(cplayer, game.players)
        return np.array(out)
    
    def output_size(self, game):
        return game.shz * game.players * exh.onehot_len