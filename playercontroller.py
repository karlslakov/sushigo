import numpy as np
import gch
import random
from trueskill import Rating

class player_controller:
    def __init__(self, name="unamed_pc"):
        self.name = name
    def get_output(self, game, player):
        raise Exception()

class agent_player_controller(player_controller):
    def __init__(self, agent):
        super().__init__(agent.name)
        self.agent = agent

    def get_output(self, game, player):
        if game.train_controller and random.random() < game.train_controller.epsilon:
            return np.random.rand(gch.output_size)
        else:
            return self.agent.predict(game.curr_features[player])

class random_player_controller(player_controller):
    def __init__(self, name="unamed_rpc"):
        super().__init__(name)

    def get_output(self, game, player):
        return np.random.rand(gch.output_size)

class human_player_controller(player_controller):
    def __init__(self, name="unamed_hpc"):
        super().__init__(name)

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

class ranked_player_controller(player_controller):
    def __init__(self, pc, rating=Rating()):
        super().__init__(pc.name)
        self.pc = pc
        self.rating = rating
    
    def get_output(self, game, player):
        return self.pc.get_output(game, player)