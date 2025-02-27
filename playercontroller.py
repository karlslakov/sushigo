import numpy as np
import gch
import random

class player_controller:
    def get_output(self, game, player):
        raise Exception()

class agent_player_controller(player_controller):
    def __init__(self, agent):
        self.agent = agent
    def get_output(self, game, player):
        if game.train_controller and random.random() < game.train_controller.epsilon:
            return np.random.rand(self.agent.output_size)
        else:
            return self.agent.predict(game.curr_features[player])


class random_player_controller(player_controller):
    def get_output(self, game, player):
        return np.random.rand(gch.output_size)

class human_player_controller(player_controller):
    def get_output(self, game, player):
        hand = []
        for i in np.arange(gch.onehot_len)[self.curr_round_hands[player] != 0]:
            hand.append([exh.to_card(i)] * int(self.curr_round_hands[player][i]))
        print("you see {}".format(hand))
        print("you have {}".format(self.selection_ordered[player]))
        while True:
            s = input("what do u want boui? ")
            if s not in card_counts or self.curr_round_hands[player][exh.to_int(s)] <= 0:
                print("no?")
                continue
            return np.array(exh.to_onehot_embedding(s))