import signal
import sys
import uuid
from traincontroller import train_controller
from config import all_features
import playercontroller as pc
from game import Game

def end_loop(agent, save):
    print("finished model training, saving stats")
    if save:
        agent.model.save(save)

def train_loop(players, agent, iters=101, save_model_freq=10, benchmark_folder=None, save=None):
    tc = train_controller(agent)
    features = all_features
    agent_pc = pc.ranked_player_controller(pc.agent_player_controller(agent))
    player_controllers = [agent_pc] * players

    game_train = Game(players, features, player_controllers, tc)
    game_eval = Game(players, features, player_controllers, None)

    def save_on_exit(sig, frame):
        end_loop(agent, save)
        sys.exit(0)

    signal.signal(signal.SIGINT, save_on_exit)

    for i in range(iters):
        print("iter %d" % i)
        game_train.play_sim_game()
        if game_train.train_controller.epsilon > 0.01:
            game_train.train_controller.epsilon -= 0.0005
        if benchmark_folder and (i % save_model_freq == 0 or i == iters - 1):
            agent.model.save("{}/benchmark_{}_iters.h5".format(benchmark_folder, i+1))

    end_loop(agent, save)
