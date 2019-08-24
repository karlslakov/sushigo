import math
from playercontroller import agent_player_controller

# use SME from http://www.tckerrigan.com/Misc/Multiplayer_Elo/

def update_agents(game, agent, benchmarks):
    agent_pc = agent_player_controller(agent)


  
# from https://www.geeksforgeeks.org/elo-rating-algorithm/
def get_probability(rating1, rating2): 
    return 1.0 * 1.0 / (1 + 1.0 * math.pow(10, 1.0 * (rating1 - rating2) / 400)) 
  
def update_elo(Ra, Rb, K, d): 
    # To calculate the Winning 
    # Probability of Player B 
    Pb = get_probability(Ra, Rb) 
  
    # To calculate the Winning 
    # Probability of Player A 
    Pa = get_probability(Rb, Ra) 
  
    # Case -1 When Player A wins 
    # Updating the Elo Ratings 
    if (d == 1) : 
        Ra = Ra + K * (1 - Pa) 
        Rb = Rb + K * (0 - Pb) 
      
  
    # Case -2 When Player B wins 
    # Updating the Elo Ratings 
    else : 
        Ra = Ra + K * (0 - Pa) 
        Rb = Rb + K * (1 - Pb) 

    return Ra, Rb
  