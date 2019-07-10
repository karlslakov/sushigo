import gch
import numpy as np

print(gch.calculate_final_score([['m1', 'p', 'm2'], ['m2', 'm3', 't'], ['m1', 'm1', 'm3', 'p', 'p']], True))

print(gch.punish_invalid(np.random.rand(8 + 4*7), 8, 6, False))
print(gch.punish_invalid(np.random.rand(8 + 4*7), 8, 6, True))