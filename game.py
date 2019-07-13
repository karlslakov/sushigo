import random
from agent import agent
import feature_extractors.extractor_helpers as exh
import numpy as np
from constants import card_counts, nigiri_scores, maki_counts
import gch 
from benchmarkagent import decide

base_deck = []



class Game:
    def __init__(self, players, feature_extractors):
        self.players = int(players)
        self.player_controllers = []
        for _ in range(players):
            self.player_controllers.append('agent')
        self.shz = 12 - self.players
        self.feature_extractors = feature_extractors
        self.agent = agent(self)
        self.epsilon = 0.4

    def get_base_deck(self):
        if len(base_deck) != 0:
            return base_deck.copy()
        for x in card_counts.keys():
            for _ in range(card_counts[x]):
                base_deck.append(x)
        return base_deck.copy()

    def watch_print(self, watch, text):
        if watch:
            print(text)
    
    def watch_wait(self, watch):
        if watch:
            input("press enter to continue")

    def start_sim_game(self, watch=False):
        print("Starting sim game with players = %d" % self.players)
        self.deck = self.get_base_deck()
        self.first_picks = np.zeros(exh.onehot_len)
        random.shuffle(self.deck)

        self.player_selected = []
        for _ in range(self.players):
            self.player_selected.append([])
        
        self.in_round_card = 0
        self.true_scores = np.zeros(self.players)
        self.round = 0
        self.watch_print(watch, "starting round 0")
        self.play_sim_round(watch)
        self.watch_print(watch, "end round 0")
        self.watch_print(watch, self.true_scores)
        self.watch_print(watch, self.player_selected)
        self.watch_print(watch, self.true_scores)
        self.watch_print(watch, self.player_selected)

        print("--- stats ---")
        print(self.true_scores)
        self.watch_wait(watch)

        return np.mean(self.true_scores), np.max(self.true_scores), np.min(self.true_scores)


    
    def get_output_for_player(self, player):
        if self.player_controllers[player] == 'agent':
            if random.random() < self.epsilon:
                self.outputs[player] = np.random.rand(self.agent.get_output_size())
            else:
                self.outputs[player] = self.agent.predict(self.curr_features[player])
        else:
            self.outputs[player] = np.random.rand(self.agent.get_output_size())
    
    def execute_action(self, action, player):
        index = action
        self.player_selected[player].append(self.curr_round_hands[player][index])
        del self.curr_round_hands[player][index]

    def is_round_over(self):
        return self.in_round_card == self.shz - 1

    def is_game_over(self):
        return self.is_round_over() # and self.round >= 2

    def play_sim_round(self, watch=False):
        offset = self.round * self.players
        self.curr_round_hands = []
        self.curr_features = []
        self.deltas = np.zeros(self.players)
        self.actions = []
        self.outputs = []
        self.temp_scores = np.array(self.true_scores)

        for i in range(self.players):
            self.curr_round_hands.append(self.deck[(offset + i) * self.shz : (offset + i + 1) * self.shz])
        
        
        for player in range(self.players):
            self.curr_features.append(exh.extract_features(self.feature_extractors, player, self))
            self.actions.append((0, False, 0))
            self.outputs.append([])

        for t in range(self.shz): 
            self.in_round_card = t                       
            for player in range(self.players):
                self.watch_print(watch, "player {} sees {}".format(player, self.curr_round_hands[player]))
                self.watch_print(watch, self.curr_features[player])
                self.get_output_for_player(player)
                self.watch_print(watch, self.outputs[player])
                self.actions[player] = gch.parse_output(self.outputs[player], self.shz, self.shz - t, self.player_selected[player])
                self.watch_print(watch, "action: {}".format(self.actions[player]))
                self.watch_wait(watch)
                self.execute_action(self.actions[player], player)

            if self.is_round_over():
                scores = gch.calculate_final_score(self.player_selected, self.is_game_over())
                self.deltas = scores
                self.true_scores += scores 
                self.temp_scores = self.true_scores

            for player in range(self.players):
                if not self.is_round_over():
                    old = self.temp_scores[player]                
                    self.temp_scores[player] = self.true_scores[player] + gch.calculate_intermediate_score(self.player_selected[player])
                    self.deltas[player] = self.temp_scores[player] - old
                new_features = exh.extract_features(self.feature_extractors, player, self)
                reward = gch.get_reward(self.true_scores, self.temp_scores, self.deltas, self.is_game_over(), player)
                self.agent.step(
                    self.curr_features[player], 
                    self.outputs[player],
                    reward, 
                    new_features,
                    self.is_game_over())
                self.agent.replay(self.agent.states, self.agent.targets)
                
                self.curr_features[player] = new_features
            
            splayer_hand = self.curr_round_hands[0]
            for p in range(self.players - 1):
                # TODO
                self.curr_round_hands[p] = self.curr_round_hands[p + 1]
            self.curr_round_hands[-1] = splayer_hand
        
        for p in range(self.players):
            self.player_selected[p] = [x for x in self.player_selected[p] if x == 'p']


    def start_irl_game(self):
        sh = input("Input hand: ")
        
  