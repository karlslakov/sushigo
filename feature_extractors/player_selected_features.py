import feature_extractors.extractor_helpers as exh
import gch
import numpy as np

class player_selected_features:
    def extract(self, player, game):
        cplayer = player
        out = []
        for i in range(game.players):
            out.append(np.zeros(exh.onehot_len))
            for c in game.player_selected[cplayer]:
                out[i] += exh.to_onehot_embedding(c)
            cplayer = gch.get_clockwise_player(cplayer, game.players)
        return np.array(out)
    
    def output_size(self, game):
        return game.players * exh.onehot_len