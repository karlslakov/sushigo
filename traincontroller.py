import numpy as np
import random
import feature_extractors.extractor_helpers as exh
import gch
import keras.backend as K
# not liking this, will update as training option requirements become more clear
class train_controller:
    def __init__(self, agent, epsilon=0.6, gamma=0.9, learning_rate=0.01, batch_size=1024, replay_memory_size=40000):
        self.agent = agent
        self.epsilon = epsilon
        self.gamma = gamma
        self.learning_rate = learning_rate
        K.set_value(agent.model.optimizer.lr, learning_rate)
        self.memory = []
        self.batch_size = batch_size
        self.replay_memory_size = replay_memory_size

    def train(self, game, player, reward, next_state, next_invalids):
        self.step(
            game.curr_features[player], 
            np.argmax(game.outputs[player]),
            reward, 
            next_state,
            next_invalids,
            game.is_round_over())
        if game.is_game_over():
            self.replay(self.memory)

    def remember(self, state, action, reward, next_state, invalid_outputs, done):
        self.memory.append((state, action, reward, next_state, invalid_outputs, done))
        if len(self.memory) > self.replay_memory_size:
            self.memory.pop(0)

    def step(self, state, action, reward, next_state, invalid_outputs, done):
        self.remember(state, action, reward, next_state, invalid_outputs, done)
        # self.train_once(state, action, reward, next_state, invalid_outputs, done)

    def replay(self, memory):
        replay = memory
        if len(memory) > self.batch_size:
            replay = random.sample(memory, self.batch_size)
        
        inputs = np.zeros((len(replay), self.agent.input_size))
        outputs = np.zeros((len(replay), gch.output_size))
        i = 0
        for state, action, reward, next_state, invalid_outputs, done in replay:
            state, target = self.get_xy(state, action, reward, next_state, invalid_outputs, done)
            inputs[i] = state
            outputs[i] = target
            i += 1
        self.agent.model.fit(inputs, outputs, epochs=1, verbose=0)
        
    def get_xy(self, state, action, reward, next_state, invalid_outputs, done):
        r = reward
        if not done:
            next_q = self.agent.model.predict(next_state.reshape((1, self.agent.input_size)))[0]
            next_q[invalid_outputs == 1] = float("-inf")
            r = r + self.gamma * np.amax(next_q)
        target = self.agent.model.predict(state.reshape((1, self.agent.input_size)))[0]
        target[action] = r
        return state, target

    def train_once(self, state, action, reward, next_state, invalid_outputs, done):        
        state, target = self.get_xy(state, action, reward, next_state, invalid_outputs, done)
        self.agent.model.fit(state.reshape((1, self.agent.input_size)), target.reshape((1, gch.output_size)), epochs=1, verbose=0)