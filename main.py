from game import Game
import argparse
from feature_extractors import player_hand_features, game_metadata_features, player_selected_features
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

def get_features():
    return [
        player_hand_features.player_hand_features(),
        game_metadata_features.game_metadata_features(),
        player_selected_features.player_selected_features(),
    ]

def train_loop(g, iters, save=None):
    avgs = []
    fps = []
    ratios = []
    for i in range(iters):
        print("iter %d" % i)
        avg, max, min = g.start_sim_game()
        ratio = g.true_scores[0] / sum(g.true_scores)
        print(avg)
        print(ratio)
        ratios.append(ratio)
        avgs.append(avg)
        fps.append(g.first_picks)
        if g.epsilon > 0.001:
            g.epsilon /= 1.1

    plt.plot(range(iters), avgs)
    plt.show()

    plt.plot(range(iters), ratios)
    plt.show()

    if save:
        g.agent.model.save(save)
    np.save("fpstats", fps)
    np.save("avgs", avgs)
    np.save("ratios", ratios)

def watch(game):
    g.agent.model = load_model("newmodel.h5")
    print(g.agent.model.predict(np.random.rand(g.agent.input_size).reshape((1, g.agent.input_size))))
    print(g.agent.model.predict(np.random.rand(g.agent.input_size).reshape((1, g.agent.input_size))))
    g.epsilon = 0
    g.start_sim_game(watch=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--players', type=int, required=True)
    parser.add_argument('-w','--watch', type=bool, required=False)
    parser.add_argument('-i', '--iters', type=int, required=False)
    parser.add_argument('-s', '--save', type=str, required=False)

    io_args = parser.parse_args()
    players = io_args.players

    features = get_features()

    g = Game(players, features)

    if not io_args.watch:
        train_loop(g, io_args.iters, io_args.save)
    else:
        watch(g)