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
        self.states = []
        self.targets = []
        self.gamma = 0.9
        self.learning_rate = 0.01
        self.output_size = gch.output_size
        self.model = self.create_model()

    def create_model(self, weights=None):
        model = Sequential()
        model.add(Dense(output_dim=120, activation='relu', input_dim=self.input_size))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=self.output_size, activation='linear'))
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if weights:
            model.load_weights(weights)
        return model
    
    def predict(self, features):
        return self.model.predict(features.reshape((1, self.input_size)))[0]

    def remember(self, state, target_f):
        self.states.append(state)
        self.targets.append(target_f)

    def step(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, self.input_size)))[0])
        target_f = self.model.predict(state.reshape((1, self.input_size)))
        target_f[0][np.argmax(action)] = target
        self.remember(state, target_f.flatten())

    def replay(self, states, targets):
        x = np.array(states).reshape((len(states), self.input_size))
        y = np.array(targets).reshape(len(targets), self.output_size)

        bsize = min(len(states), 100)
        c = np.random.choice(len(states), bsize, replace=False)
        x = x[c]
        y = y[c]

        self.model.fit(x, y, epochs=1, verbose=0)