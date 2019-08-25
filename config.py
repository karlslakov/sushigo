from feature_extractors import ( 
    player_hand_features,
    game_metadata_features, 
    player_selected_features, 
    strategy_helper_features,
    discard_features,
    score_features,
)

all_features = [
    player_hand_features.player_hand_features(),
    game_metadata_features.game_metadata_features(),
    player_selected_features.player_selected_features(),
    strategy_helper_features.strategy_helper_features(),
    discard_features.discard_features(),
    score_features.score_features(),
]