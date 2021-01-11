from game import Game
import argparse
from feature_extractors import ( 
    player_hand_features,
    game_metadata_features, 
    player_selected_features, 
    strategy_helper_features,
    discard_features,
    score_features,
)
import numpy as np
import signal
import sys
import playercontroller as pc
import agent as ag
import traincontroller as tc
import torch
from torch.utils.tensorboard import SummaryWriter
import copy


def get_features():
    return [
        player_hand_features.player_hand_features(),
        game_metadata_features.game_metadata_features(),
        player_selected_features.player_selected_features(),
        strategy_helper_features.strategy_helper_features(),
        # discard_features.discard_features(),
        score_features.score_features(),
    ]

def end_loop(iters, g, save):
    print("finished model training, saving stats")
    if save and g.train_controller:
        torch.save(g.train_controller.agent.model, save)

def train_loop(g, iters, save=None):
    def save_on_exit(sig, frame):
        end_loop(iters, g, save)
        sys.exit(0)
    signal.signal(signal.SIGINT, save_on_exit)
    torch.autograd.set_detect_anomaly(True)
    writer = SummaryWriter(comment="train_v_rands")

    MODEL_CHECKPOINT_DIR = "model_checkpoints/"


    

    def checkpoint(iter):
        path = "%s_model%d" % (MODEL_CHECKPOINT_DIR, iter)
        torch.save(g.train_controller.agent.model, path)
        for i in range(1, g.players):
            if isinstance(g.player_controllers[i], pc.agent_player_controller):
                g.player_controllers[i].agent.model = copy.deepcopy(g.train_controller.agent.model)

    for i in range(iters):
        if i % 250 == 0:
            checkpoint(i)
            # pass
        print("iter %d" % i)
        g.play_sim_game()

        ratio = g.true_scores[0] / sum(g.true_scores)
        argsorted = np.argsort(g.true_scores)
        places = argsorted.tolist()
        places.reverse()
        place = places.index(0)
        avg_score = g.true_scores.mean()
        writer.add_scalar("Stats/MeanScore", avg_score, i)
        writer.add_scalar("Stats/Ratio", ratio, i)
        writer.add_scalar("Stats/Place", place, i)

        
        # for name, param in g.player_controllers[1].agent.model.named_parameters():
            # y = param.data
            # break
        #for name, param in g.player_controllers[1].agent.model.named_parameters():
         #   x = param.data
          #  break
        
        #print(torch.all(torch.eq(x,y)))
        
        

    end_loop(iters, g, save)


def watch(game):
    np.set_printoptions(precision=2)
    g.play_sim_game_watched()

def play(game):
    g.play_sim_game_watched(verbose=0)

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
        # pcs.append(pc.random_player_controller())
        pcs.append(pc.agent_player_controller(copy.deepcopy(agent), trainable=(i == 0)))
    pcs[0] = pc.agent_player_controller(agent)
    
    if args.irl_type == 'hvsall':
        pcs[0] = pc.human_player_controller()
    elif args.irl_type == 'cvsall':
        raise Exception("cvsall not implemented")
    elif args.eval:
        for i in range(1, args.players):
            pcs[i] = pc.random_player_controller()
    
    return pcs

def get_agent(features, players, args):
    agent = ag.agent(features, players, args.load)
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
    parser.add_argument('--irl_type', type=str, required=False)
    parser.add_argument('--benchmark', type=str, required=False)


    io_args = parser.parse_args()
    players = io_args.players

    features = get_features()
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
            eval_model(g, io_args.iters)
        else:
            train_loop(g, io_args.iters, io_args.save)
    else:
        watch(g)
