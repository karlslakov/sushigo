from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Input
from keras.layers.core import Dense, Dropout
import random
import numpy as np
import feature_extractors.extractor_helpers as exh
import gch

class agent:
    def __init__(self, game):
        self.players = game.players
        self.shz = game.shz
        self.feature_extractors = game.feature_extractors
        self.input_size = exh.get_input_size(self.feature_extractors, game) 
        self.memory = []
        self.gamma = 0.9
        self.learning_rate = 0.01
        self.output_size = gch.output_size
        self.model = self.create_model()

    def create_model(self, weights=None):
        model = Sequential()
        model.add(Dense(output_dim=120, activation='relu', input_dim=self.input_size))
        model.add(Dense(output_dim=120, activation='relu'))
        model.add(Dense(output_dim=120, activation='relu'))
        model.add(Dense(output_dim=self.output_size, activation='linear'))
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if weights:
            model.load_weights(weights)
        return model
    
    def predict(self, features):
        return self.model.predict(features.reshape((1, self.input_size)))[0]

    def remember(self, state, action, reward, next_state, invalid_outputs, done):
        self.memory.append((state, action, reward, next_state, invalid_outputs, done))
        if len(self.memory) > 50000:
            self.memory.pop(0)

    def step(self, state, action, reward, next_state, invalid_outputs, done):
        self.remember(state, action, reward, next_state, invalid_outputs, done)
        self.train_once(state, action, reward, next_state, invalid_outputs, done)

    def replay(self, memory):
        replay = memory
        if len(memory) > 1000:
            replay = random.sample(memory, 1000)
        
        inputs = np.zeros((len(replay), self.input_size))
        outputs = np.zeros((len(replay), self.output_size))
        i = 0
        for state, action, reward, next_state, invalid_outputs, done in replay:
            state, target = self.get_xy(state, action, reward, next_state, invalid_outputs, done)
            inputs[i] = state
            outputs[i] = target
            i += 1
        self.model.fit(inputs, outputs, epochs=5, verbose=0)
        
    def get_xy(self, state, action, reward, next_state, invalid_outputs, done):
        r = reward
        if not done:
            r = r + self.gamma * np.amax(self.model.predict(next_state.reshape((1, self.input_size)))[0])
        target = self.model.predict(state.reshape((1, self.input_size)))[0]
        target[action] = r
        target[invalid_outputs == 1] = -10
        return state, target

    def train_once(self, state, action, reward, next_state, invalid_outputs, done):        
        state, target = self.get_xy(state, action, reward, next_state, invalid_outputs, done)
        self.model.fit(state.reshape((1, self.input_size)), target.reshape((1, self.output_size)), epochs=1, verbose=0)
