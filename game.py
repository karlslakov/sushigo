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
        random.shuffle(self.deck)

        self.player_selected = []
        self.selection_ordered = []
        for _ in range(self.players):
            self.player_selected.append(np.zeros(exh.onehot_len))
            self.selection_ordered.append([])
        
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
        self.agent.replay(self.agent.memory)
        self.watch_wait(watch)

        return np.mean(self.true_scores), np.max(self.true_scores), np.min(self.true_scores)

    def get_output_for_player(self, player):
        self.invalid_outputs[player] = gch.get_invalid_outputs(self.curr_round_hands[player], self.player_selected[player])
        if self.player_controllers[player] == 'agent':
            if random.random() < self.epsilon:
                self.unfiltered_outputs[player] = np.random.rand(self.agent.output_size)
            else:
                self.unfiltered_outputs[player] = self.agent.predict(self.curr_features[player])
        else:
            self.unfiltered_outputs[player] = np.random.rand(self.agent.output_size)
        self.outputs[player] = self.unfiltered_outputs[player].copy()
        self.outputs[player][self.invalid_outputs[player] == 1] = float("-inf")
    
    def execute_action(self, action, player):
        self.player_selected[player][action] += 1
        self.curr_round_hands[player][action] -= 1
        self.selection_ordered[player].append(exh.to_card(action))

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
        self.unfiltered_outputs = []
        self.temp_scores = np.array(self.true_scores)
        self.invalid_outputs = []

        for i in range(self.players):
            hand = self.deck[(offset + i) * self.shz : (offset + i + 1) * self.shz]
            self.curr_round_hands.append(exh.to_counts(hand))
        
        for player in range(self.players):
            self.curr_features.append(exh.extract_features(self.feature_extractors, player, self))
            self.actions.append((0, False, 0))
            self.outputs.append([])
            self.unfiltered_outputs.append([])
            self.invalid_outputs.append([])

        for t in range(self.shz): 
            self.in_round_card = t                       
            for player in range(self.players):
                self.watch_print(watch, "player {} sees {}".format(player, self.curr_round_hands[player]))
                self.watch_print(watch, self.curr_features[player])
                self.get_output_for_player(player)
                self.watch_print(watch, self.unfiltered_outputs[player])
                self.actions[player] = gch.parse_output(self.outputs[player], self.curr_round_hands[player], self.player_selected[player])
                self.watch_print(watch, "action: {}".format(self.actions[player]))
                self.watch_wait(watch)
                self.execute_action(self.actions[player], player)

            if self.is_round_over():
                scores = gch.calculate_final_score(self.selection_ordered, self.is_game_over())
                self.deltas = scores
                self.true_scores += scores 
                self.temp_scores = self.true_scores

            splayer_hand = self.curr_round_hands[0]
            for p in range(self.players - 1):
                # TODO
                self.curr_round_hands[p] = self.curr_round_hands[p + 1]
            self.curr_round_hands[-1] = splayer_hand

            for player in range(self.players):
                if not self.is_round_over():
                    old = self.temp_scores[player]                
                    self.temp_scores[player] = self.true_scores[player] + gch.calculate_intermediate_score(self.selection_ordered[player])
                    self.deltas[player] = self.temp_scores[player] - old
               
                reward = gch.get_reward(self.true_scores, self.temp_scores, self.is_game_over(), player)
                new_features = exh.extract_features(self.feature_extractors, player, self)

                self.agent.step(
                    self.curr_features[player], 
                    np.argmax(self.outputs[player]),
                    reward, 
                    new_features,
                    self.invalid_outputs[player],
                    self.is_game_over())
                
                self.curr_features[player] = new_features

    def start_irl_game(self):
        sh = input("Input hand: ")
        
  