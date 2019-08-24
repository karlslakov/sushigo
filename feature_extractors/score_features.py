import numpy as np
import feature_extractors.extractor_helpers as exh
import gch

class score_features:
    def extract(self, player, game):
        cplayer = player
        out = []
        for i in range(game.players):
            out.append(game.true_scores[cplayer])
            out.append(game.temp_scores[cplayer])
            cplayer = gch.get_next_player(cplayer, game)
        
        return np.array(out)
    
    def output_size(self, players):
        return players * 2