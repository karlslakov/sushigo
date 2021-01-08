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
        self.ll_invalids = []
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

    def run(self, features, invalids):
        return self.model(torch.tensor(features, dtype=torch.float).to('cuda'),
            torch.tensor(1 - invalids, dtype=torch.bool).to('cuda')).cpu()

    def remember(self, ll, r, channel):
        while len(self.log_likelyhoods) <= channel:
            self.log_likelyhoods.append([])
            self.rewards.append([])
            
        # print(r)
        self.log_likelyhoods[channel].append(ll)
        self.rewards[channel].append(r)

    def step_train(self, train_controller):
        lls_pre = []
        r_pre = []
        for i in range(len(self.log_likelyhoods)):
            lls_pre.append(torch.vstack(self.log_likelyhoods[i]))
            r_pre.extend(train_controller.propogate_reward(self.rewards[i]))

        # print(r_pre)
        lls = torch.vstack(lls_pre)
        r = torch.vstack(r_pre)

        # print(r)
        r -= r.mean()
        
        if r.std().item() != 0:
            r /= r.std()
        # print(r)

        loss = -(r.unsqueeze(1) * lls).mean()
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()

        self.log_likelyhoods = []
        self.rewards = []
        self.ll_invalids = []