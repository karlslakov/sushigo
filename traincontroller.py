import numpy as np
import torch

INVALID_ACTION_REWARD = -10

class train_controller:
    def __init__(self, agent):
        self.agent = agent

    def register_turn(self, game, player, invalid_action_chose):
        reward = 0
        if game.is_game_over():
            argsorted = np.argsort(game.true_scores)
            places = argsorted.tolist()
            place = places.index(player)
            reward = game.true_scores[player] + 35 * place
        else:
            reward = game.deltas[player]
        
        self.agent.remember(game.outputs[player][game.origs[player]], \
            INVALID_ACTION_REWARD if invalid_action_chose else reward)

        if game.is_game_over():
            self.agent.step_train(self)

    def propogate_reward(self, rewards):
        cumsum = 0
        out = torch.zeros(len(rewards))
        for t in reversed(range(0, len(rewards))):
            if rewards[t] == INVALID_ACTION_REWARD:
                out[t] = INVALID_ACTION_REWARD
            else:
                # cumsum = cumsum * 0.9 + rewards[t]
                # out[t] = cumsum
                out[t] = -INVALID_ACTION_REWARD
        return out
