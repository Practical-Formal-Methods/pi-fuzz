SEARCH_BUDGET = 100
COV_DIST_THOLD = 3  # euclidean
MM_MUT_MAGNITUDE = 2
LA_MUT_MAGNITUDE = 3
POOL_POP_MUT = 3
POOL_BUDGET = 500
DEVIATION_DEPTH = 3
DELTA = 1  # quantitative oracle ensures there is a flaw

import numpy as np
RANDOM_SEED = 123
RNG = np.random.default_rng(RANDOM_SEED)  # IMPORTANT FOR REPRODUCABILITY
