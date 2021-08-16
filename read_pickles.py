import pickle
from os import listdir
from os.path import isfile, join
from fuzz_utils import plot_rq3_time_cloud

env_iden = "linetrack"
fuzz_type = "gbox"
rand_seeds = [1, 2, 3, 4, 5, 6, 7, 8]

pool_summ_gb, pool_summ_bb = [], []
for f in listdir("linetrack_results"):
    fname = "%s" % (env_iden)
    if isfile(join("linetrack_results", f)) and fname in f:
        fuzz_r = pickle.load(open(join("linetrack_results", f), "rb"))
        pop_summ, pool, tot_warns_mm_e, tot_warns_mm_h, num_cycles, total_trials = fuzz_r
        print(f, len(pool))
        if "gbox" in f:
            pool_summ_gb.append(pop_summ)
        else:
            pool_summ_bb.append(pop_summ)

                
plot_rq3_time_cloud(pool_summ_gb, pool_summ_bb)

