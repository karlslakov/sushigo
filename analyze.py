import numpy as np
import feature_extractors.extractor_helpers as exh


fpstats = np.load('fpstats.npy')

for game in fpstats:
    a = np.argmax(game)
    onehot = np.zeros(exh.onehot_len)
    onehot[a] = 1
    print(exh.from_onehot_embedding(onehot))