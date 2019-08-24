import numpy as np
# not liking this, will update as training option requirements become more clear
class train_controller:
    def __init__(self, agent, epsilon=0.6):
        self.agent = agent
        self.epsilon = epsilon

    def train(self, game, player, reward, next_state, next_invalids):
        self.agent.step(
            game.curr_features[player], 
            np.argmax(game.outputs[player]),
            reward, 
            next_state,
            next_invalids,
            game.is_round_over())
        self.agent.replay(self.agent.memory)
