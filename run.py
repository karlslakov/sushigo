from game import Game
import argparse
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
import playercontroller as pc
import agent as ag
import traincontroller as tc
import feature_extractors.extractor_helpers as exh
import gch
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Input
from keras.layers.core import Dense, Dropout
from config import all_features
from training import train_loop
import eval

def watch(game):
    g.play_sim_game_watched()

def play(game):
    g.play_sim_game_watched(verbose=1)

def calibrate_elo(game, folder, iters=50):
    input_size = exh.get_input_size( game.feature_extractors, game.players)
    agents = eval.get_agents(folder, input_size)
    ratings = eval.calibrate_agent_elo(game, agents)
    print(ratings)

def eval_model(game, iters = 50): 
    places_stats = []
    ratios = []
    
    for i in range(iters):
        print("iter %d" % i)
        g.play_sim_game()
        ratio = g.true_scores[0] / sum(g.true_scores)
        argsorted = np.argsort(g.true_scores)
        places = argsorted.tolist()
        places.reverse()
        place = places.index(0)
        places_stats.append(place)
        ratios.append(ratio)
    
    places_stats = np.array(places_stats)
    ratios = np.array(ratios)

    print("ratios:")
    print(ratios)
    print("places")
    print(places_stats)
    print("avg ratio: {}".format(np.mean(ratios)))
    print("avg place: {}".format(np.mean(places_stats)))
    print("1st places: {}".format(np.count_nonzero(places_stats == 0)))
    print("2nd places: {}".format(np.count_nonzero(places_stats == 1)))
    print("3rd places: {}".format(np.count_nonzero(places_stats == 2)))
    print("4th places: {}".format(np.count_nonzero(places_stats == 3)))

def get_player_controllers(args, agent):
    pcs = []
    for i in range(args.players):
        pcs.append(pc.agent_player_controller(agent))
    
    if args.irl_type == 'hvsall':
        pcs[0] = pc.human_player_controller()
    elif args.irl_type == 'cvsall':
        raise Exception("cvsall not implemented")
            
    return pcs

def create_model(input_size, output_size):
    model = Sequential()
    model.add(Dense(output_dim=120, activation='relu', input_dim=input_size))
    model.add(Dense(output_dim=120, activation='relu'))
    model.add(Dense(output_dim=120, activation='relu'))
    model.add(Dense(output_dim=output_size, activation='linear'))
    opt = RMSprop()
    model.compile(loss='mse', optimizer=opt)
    
    return model

def get_agent(features, players, args):
    input_size = exh.get_input_size(features, players)
    model = load_model(args.load) if args.load else create_model(input_size, gch.output_size)
    agent = ag.agent(model, input_size)
    return agent


def get_train_controller(agent, args):
    if args.eval or args.watch or args.irl_type != None:
        return None
    return tc.train_controller(agent)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--players', type=int, required=True)
    parser.add_argument('-w','--watch', action="store_const", const=True, required=False)
    parser.add_argument('-i', '--iters', type=int, required=False)
    parser.add_argument('-s', '--save', type=str, required=False)
    parser.add_argument('-l', '--load', type=str, required=False)
    parser.add_argument('-e', '--eval', action="store_const", const=True, required=False)
    parser.add_argument('--benchmark_folder', type=str, required=False)
    parser.add_argument('--irl_type', type=str, required=False)
    parser.add_argument('--benchmark', type=str, required=False)

    io_args = parser.parse_args()
    players = io_args.players

    features = all_features
    agent = get_agent(features, players, io_args)
    player_controllers = get_player_controllers(io_args, agent)
    train_controller = get_train_controller(agent, io_args)
    g = Game(players, features, player_controllers, train_controller)

    if io_args.irl_type == 'cpuvsall':
        g.start_irl_game()
    elif io_args.irl_type == 'hvsall':
        play(g)
    elif not io_args.watch:
        if io_args.eval:
            calibrate_elo(g, io_args.benchmark_folder, io_args.iters)
        else:
            train_loop(players, agent, iters=io_args.iters, save=io_args.save, benchmark_folder=io_args.benchmark_folder)
    else:
        watch(g)
