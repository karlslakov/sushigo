import math
from playercontroller import ranked_player_controller, random_player_controller, agent_player_controller
import random
import numpy as np
import trueskill
import os
from agent import agent
from keras.models import load_model


def calibrate_agent_elo(game, agents, eval_epochs=100):
    players = game.players

    agent_controllers = [ranked_player_controller(agent_player_controller(x)) for x in agents]
    
    assert len(agent_controllers) >= players
    assert game.train_controller == None

    for i in range(eval_epochs):
        game.player_controllers = random.sample(agent_controllers, k=players)
        game.play_sim_game()

        argsorted = np.argsort(game.true_scores)
        places = argsorted.tolist()
        places.reverse()

        teams = [(x.rating,) for x in game.player_controllers]
        new_ratings = trueskill.rate(teams, places)

        for i in range(players):
            game.player_controllers[i].rating = new_ratings[i][0]
        
    return [(x.pc.agent.name, x.rating.mu, x.rating.sigma) for x in agent_controllers]
            
def get_agents(folder, input_size):
    agents = []
    for filename in os.listdir(folder):
        if filename.endswith(".h5"):
            model = load_model(os.path.join(folder, filename))
            agents.append(agent(model, input_size, name=filename))
    return agents

        


