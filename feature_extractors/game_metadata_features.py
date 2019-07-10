import numpy as np

class game_metadata_features:
    def extract(self, player, game):
        return np.array([game.players, game.round, game.in_round_card])
    
    def output_size(self, game):
        return 3