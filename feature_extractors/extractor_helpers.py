import numpy as np
from constants import card_counts

card_to_int = dict((c, i) for i, c in enumerate(card_counts))
int_to_card = dict((i, c) for i, c in enumerate(card_counts))
onehot_len = len(card_counts)
nocard_onehot = np.zeros(onehot_len)


def get_input_size(feature_extractors, game):
    return np.sum([f.output_size(game) for f in feature_extractors])

def extract_features(feature_extractors, player, game):
    return np.concatenate([f.extract(player, game).flatten() for f in feature_extractors])

def to_onehot_embedding(card):
    onehot = [0 for _ in range(onehot_len)]
    onehot[card_to_int[card]] = 1
    return onehot

def from_onehot_embedding(onehot):
    return int_to_card[np.argmax(onehot)]

def to_int(card):
    return card_to_int[card]

def to_card(i):
    return int_to_card[i]
