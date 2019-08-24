import numpy as np
from constants import card_counts

card_to_int = dict((c, i) for i, c in enumerate(card_counts))
int_to_card = dict((i, c) for i, c in enumerate(card_counts))
onehot_len = len(card_counts)
nocard_onehot = np.zeros(onehot_len)


def get_input_size(feature_extractors, players):
    return np.sum([f.output_size(players) for f in feature_extractors])

def extract_features(feature_extractors, player, game):
    def extract_and_assert(f):
        features = f.extract(player, game).flatten()
        assert len(features) == f.output_size(game.players), "features extracted from {} not correct size.\nExpected size: {}\nActual size:{}".format(f, f.output_size(game), len(features))
        return features
    return np.concatenate([extract_and_assert(f) for f in feature_extractors])

def to_onehot_embedding(card):
    onehot = [0 for _ in range(onehot_len)]
    onehot[card_to_int[card]] = 1
    return onehot

def to_counts(cards):
    counts = np.zeros(onehot_len)
    for c in cards:
        counts += to_onehot_embedding(c)
    return counts

def from_onehot_embedding(onehot):
    return int_to_card[np.argmax(onehot)]

def to_int(card):
    return card_to_int[card]

def to_card(i):
    return int_to_card[i]
