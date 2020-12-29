import random
import numpy as np
import feature_extractors.extractor_helpers as exh
import gch
from rlmodel import PGModel
import torch.optim as optim
import torch


class agent:
    def __init__(self, feature_extractors, players, load_model = None):
        self.players = players
        self.input_size = exh.get_input_size(feature_extractors, players) 
        self.log_likelyhoods = []
        self.rewards = []
        self.output_size = gch.output_size

        if load_model:
            self.model = torch.load(load_model)
        else:
            self.model = self.create_model()
        self.opt = optim.RMSprop(self.model.parameters(), lr=0.01)

    def create_model(self):
        model = PGModel(self.input_size, self.output_size).to('cuda')
        return model

    def run(self, features):
        return self.model(torch.tensor(features, dtype=torch.float).to('cuda')).cpu()

    def remember(self, ll, r):
        self.log_likelyhoods.append(ll)
        self.rewards.append(r)

    def step_train(self, train_controller):
        lls = torch.vstack(self.log_likelyhoods)
        r = train_controller.propogate_reward(self.rewards)

        # print(r.mean())
        r -= r.mean()
        
        if r.std().item() != 0:
            r /= r.std()
        
        
        # print (lls)
        # print(r)
        # print(r.unsqueeze(1) * lls)

        loss = -(r.unsqueeze(1) * lls).sum()
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()

        self.log_likelyhoods = []
        self.rewards = []