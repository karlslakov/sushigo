import random
from agent import agent
import feature_extractors.extractor_helpers as exh
import numpy as np
from constants import card_counts, nigiri_scores, maki_counts
import gch 

base_deck = []



class Game:
    def __init__(self, players, feature_extractors, player_controllers, train_controller):
        self.players = int(players)
        self.shz = gch.get_shz(players)
        self.player_controllers = player_controllers
        self.feature_extractors = feature_extractors
        self.train_controller = train_controller

    def get_base_deck(self):
        if len(base_deck) != 0:
            return base_deck.copy()
        for x in card_counts.keys():
            for _ in range(card_counts[x]):
                base_deck.append(x)
        return base_deck.copy()

    def init_game(self):
        print("Starting sim game with players = %d" % self.players)
        self.deck = self.get_base_deck()
        random.shuffle(self.deck)
        self.true_scores = np.zeros(self.players)
        self.player_selected = []
        self.selection_ordered = []
        self.in_round_card = 0
        self.round = 0
        for _ in range(self.players):
            self.player_selected.append(np.zeros(exh.onehot_len))
            self.selection_ordered.append([])

    def play_sim_game_watched(self, verbose=0):
        self.verbose = verbose
        self.init_game()

        print("starting round 0")
        self.play_sim_round_watched()
        print("end round 0")
        print(self.true_scores)
        print(self.player_selected)
        self.watch_wait()
        self.round = 1
        print("starting round 1")
        self.play_sim_round_watched()
        print("end round 1")
        print(self.true_scores)
        print(self.player_selected)
        self.watch_wait()
        self.round = 2
        print("starting final round")
        self.play_sim_round_watched()
        print("end final round")
        print(self.true_scores)
        self.watch_wait()

    def play_sim_game(self, round_outputs=False):
        self.init_game()
        
        self.play_sim_round()
        if round_outputs:
            print(self.true_scores)
        self.round = 1
        self.play_sim_round()
        if round_outputs:
            print(self.true_scores)
        self.round = 2
        self.play_sim_round()
        if round_outputs:
            print(self.true_scores)
       
    def get_output_for_player(self, player):
        self.unfiltered_outputs[player] = self.player_controllers[player].get_output(self, player)       
        self.outputs[player] = self.unfiltered_outputs[player].copy()
        self.outputs[player][self.invalid_outputs[player] == 1] = float("-inf")
    
    def execute_action(self, action, player):
        first = action
        self.player_selected[player][first] += 1
        self.curr_round_hands[player][first] -= 1
        self.selection_ordered[player].append(exh.to_card(first))

    def is_round_over(self):
        return self.in_round_card == self.shz - 1

    def is_game_over(self):
        return self.is_round_over() and self.round >= 2

    def init_round(self):
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
            self.invalid_outputs.append(gch.get_invalid_outputs(self.curr_round_hands[player], False))

    def rotate_player_hands(self):
        if round == 1:
            splayer_hand = self.curr_round_hands[-1]
            for p in range(self.players - 1, 0, -1):
                self.curr_round_hands[p] = self.curr_round_hands[p - 1]
            self.curr_round_hands[0] = splayer_hand
        else:
            splayer_hand = self.curr_round_hands[0]
            for p in range(self.players - 1):
                # TODO
                self.curr_round_hands[p] = self.curr_round_hands[p + 1]
            self.curr_round_hands[-1] = splayer_hand

    def update_true_scores(self):
        if self.is_round_over():
            scores = gch.calculate_final_score(self.selection_ordered, self.is_game_over())
            self.deltas = scores
            self.true_scores += scores 
            self.temp_scores = self.true_scores

    def clear_selected(self):
        for p in range(self.players):
            puddings = self.player_selected[p][exh.to_int('p')]
            self.player_selected[p] = np.zeros(exh.onehot_len)
            self.player_selected[p][exh.to_int('p')] = puddings
            self.selection_ordered[p] = [x for x in self.selection_ordered[p] if x == 'p']
    
    def prep_player_output_action(self, player):
        self.get_output_for_player(player)
        self.actions[player] = gch.parse_output(self.outputs[player], self.curr_round_hands[player], False)
        self.execute_action(self.actions[player], player)

    def end_pick_cleanup_and_train(self):
        if not self.is_round_over():
            for player in range(self.players):
                old = self.temp_scores[player]                
                self.temp_scores[player] = self.true_scores[player] + gch.calculate_intermediate_score(self.selection_ordered[player])
                self.deltas[player] = self.temp_scores[player] - old
        for player in range(self.players):
            reward = gch.get_reward(self.true_scores, self.temp_scores, self.is_game_over(), player)
            new_features = exh.extract_features(self.feature_extractors, player, self)
            next_invalids = gch.get_invalid_outputs(self.curr_round_hands[player], False)

            if self.train_controller:
                self.train_controller.train(self, player, reward, new_features, next_invalids)
            
            self.curr_features[player] = new_features
            self.invalid_outputs[player] = next_invalids

    def play_sim_round(self):
        self.init_round()        
        for self.in_round_card in range(self.shz):                       
            for player in range(self.players):
                self.prep_player_output_action(player)

            self.update_true_scores()
            self.rotate_player_hands()

            self.end_pick_cleanup_and_train()
        self.clear_selected()

    def watch_wait(self):
        input("press enter to continue")

    def play_sim_round_watched(self):
        self.init_round()        
        for self.in_round_card in range(self.shz):                   
            for player in range(self.players):
                print("player {}".format(player))
                if self.verbose == 0:
                    hand = [exh.to_card(x) for x in np.arange(gch.onehot_len)[self.curr_round_hands[player] != 0]]
                    print("player {} sees {}".format(player, hand))
                    print("player {} has {}".format(player, self.selection_ordered[player]))
                self.prep_player_output_action(player)
                
                if self.verbose == 0:
                    print(self.curr_features[player])
                    print(self.unfiltered_outputs[player])
                
                print("action: {}".format(exh.to_card(self.actions[player])))
                self.watch_wait()

            self.update_true_scores()
            self.rotate_player_hands()

            self.end_pick_cleanup_and_train()
        self.clear_selected()
        

    def start_irl_game_cpuvall(self):
        self.init_game()
        self.round = 0
        self.start_irl_round()
        self.round = 1
        self.start_irl_round()
        self.round = 2
        self.start_irl_round()

    def start_irl_round_cpuvall(self):
        self.curr_round_hands = []
        self.curr_features = [0]
        self.outputs = [0]
        self.unfiltered_outputs = [0]
        self.invalid_outputs = [0]
        for p in range(self.players):
            self.curr_round_hands.append(np.zeros(gch.onehot_len))
        
        for self.in_round_card in range(self.shz):
            print(self.curr_round_hands)
            if self.in_round_card < self.players:
                h = input("What hand do you see? ")
                cards = h.split(",")
                self.curr_round_hands[0] = exh.to_counts(cards)
            else: 
                hand = [exh.to_card(x) for x in np.arange(gch.onehot_len)[self.curr_round_hands[player] != 0]]
                print("you should see {} ".format(hand))

            self.curr_features[0] = exh.extract_features(self.feature_extractors, 0, self)
            self.invalid_outputs[0] = gch.get_invalid_outputs(self.curr_round_hands[0], False)
            self.get_output_for_player(0)
            action = gch.parse_output(self.outputs[0], self.curr_round_hands[0], False)
            print("take {}".format(exh.to_card(action)))
            self.execute_action(action, 0)

            for p in range(self.players - 1):
                player = p + 1
                while True:
                    pick = input("what did the player to your left (the one you pass to in first round) pick? ")
                    if pick in card_counts:
                        break
                    print("that card isn't real, you fool")
            
            self.rotate_player_hands()
                
                

        
  