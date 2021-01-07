import numpy as np
import torch

class train_controller:
    def __init__(self, agent, rollouts_per_train=8):
        self.agent = agent
        self.rollouts_per_train = rollouts_per_train
        self.rollouts_completed = 0

    def register_turn(self, game, player, invalids):
        reward = 0
        if game.is_game_over():
            argsorted = np.argsort(game.true_scores)
            places = argsorted.tolist()
            place = places.index(player)
            # print(game.true_scores)
            reward = place
            # print(reward)
        else:
            reward = 0
            # reward = game.deltas[player]
            # print(reward)
        
        self.agent.remember(game.outputs[player][game.actions[player]], reward, player)
        self.agent.remember_invalids(game.outputs[player], invalids)


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
