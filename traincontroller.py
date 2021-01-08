import numpy as np
import torch
import gch

class train_controller:
    def __init__(self, agent, rollouts_per_train=8):
        self.agent = agent
        self.rollouts_per_train = rollouts_per_train
        self.rollouts_completed = 0

    def register_turn(self, game, player):
        reward = 0
        if game.is_game_over():
            argsorted = np.argsort(game.true_scores)
            places = argsorted.tolist()
            place = places.index(player)
            #print(game.true_scores)
            reward = place + 1
            #print(player, reward)
        else:
            reward = 0
            # reward = game.deltas[player]
            # print(reward)
        
        index = np.where(np.arange(gch.output_size)[game.invalid_outputs[player] != 1] == game.actions[player])[0][0]
        self.agent.remember(game.outputs[player][index], reward, player)


        if game.is_game_over():
            self.rollouts_completed = self.rollouts_completed + 1
            if self.rollouts_completed >= self.rollouts_per_train:
                self.rollouts_completed = 0
                self.agent.step_train(self)

    def propogate_reward(self, rewards):
        cumsum = 0
        out = torch.zeros(len(rewards))
        for t in reversed(range(0, len(rewards))):
            if rewards[t] != 0:
                cumsum = 0
            cumsum = cumsum * 1 + rewards[t]
            out[t] = cumsum
        return out
