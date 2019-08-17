import numpy as np
import feature_extractors.extractor_helpers as exh
import gch

class discard_features:
    def extract(self, player, game):
        offset = game.round * game.players * game.shz
        return np.array(exh.to_counts(game.deck[:offset]))
    
    def output_size(self, game):
        return gch.onehot_len