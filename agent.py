from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Input
from keras.layers.core import Dense, Dropout
import random
import numpy as np
import feature_extractors.extractor_helpers as exh

class agent:
    def __init__(self, game):
        self.players = game.players
        self.shz = game.shz
        self.feature_extractors = game.feature_extractors
        self.input_size = exh.get_input_size(self.feature_extractors, game) 
        self.memory = []
        self.gamma = 0.9
        self.learning_rate = 0.01
        self.model = self.create_model()
       
    def get_output_size(self):
        return int(self.shz + self.shz * (self.shz - 1) / 2)

    def create_model(self, weights=None):
        model = Sequential()
        model.add(Dense(output_dim=120, activation='relu', input_dim=self.input_size))
        model.add(Dense(output_dim=120, activation='relu'))
        model.add(Dense(output_dim=120, activation='relu'))
        model.add(Dense(output_dim=self.get_output_size(), activation='linear'))
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if weights:
            model.load_weights(weights)
        return model
    
    def predict(self, features):
        return self.model.predict(features.reshape((1, self.input_size)))[0]

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 50000:
            self.memory.pop(0)

    def step(self, state, action, reward, next_state, done):
        self.remember(state, action, reward, next_state, done)
        self.train_once(state, action, reward, next_state, done)

    def replay(self, memory):
        replay = memory
        if len(memory) > 1000:
            replay = random.sample(memory, 1000)
        
        for state, action, reward, next_state, done in replay:
            self.train_once(state, action, reward, next_state, done)
    
    def train_once(self, state, action, reward, next_state, done):        
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, self.input_size)))[0])
        target_f = self.model.predict(state.reshape((1, self.input_size)))
        target_f[0][np.argmax(action)] = target
        self.model.fit(state.reshape((1, self.input_size)), target_f, epochs=1, verbose=0)