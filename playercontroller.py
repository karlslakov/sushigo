import numpy as np
import gch
import random
import torch
from constants import card_counts
import feature_extractors.extractor_helpers as exh

class player_controller:
    def get_output(self, game, player):
        raise Exception()

    def is_trainable(self):
        return False

class agent_player_controller(player_controller):
    def __init__(self, agent, trainable=True):
        self.agent = agent
        self.trainable = trainable
    def get_output(self, game, player):
        return self.agent.run(game.curr_features[player], game.invalid_outputs[player])
    def is_trainable(self):
        return self.trainable


class random_player_controller(player_controller):
    def get_output(self, game, player):
        return torch.log_softmax(torch.rand(np.count_nonzero(game.invalid_outputs[player] != 1)), dim=0)

class human_player_controller(player_controller):
    def get_output(self, game, player):
        hand = []
        for i in np.arange(gch.onehot_len)[game.curr_round_hands[player] != 0]:
            hand.append([exh.to_card(i)] * int(game.curr_round_hands[player][i]))
        print("you see {}".format(hand))
        print("you have {}".format(game.selection_ordered[player]))
        while True:
            s = input("what do u want boui? ")
            if s not in card_counts or game.curr_round_hands[player][exh.to_int(s)] <= 0:
                print("no?")
                continue
            
            out = np.array(exh.to_onehot_embedding(s))[game.invalid_outputs[player] != 1] * 1000
            return torch.log_softmax(torch.tensor(out, dtype=torch.float), dim=0)