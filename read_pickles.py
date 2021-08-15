import pickle
from os import listdir
from os.path import isfile, join

env_iden = "linetrack"
fuzz_type = "gbox"
rand_seeds = [1, 2, 3, 4, 5, 6, 7, 8]

ppaths = []
for f in listdir("linetrack_results"):
    for rseed in rand_seeds:
        fname = "%s_%s_%d.p"%(env_iden, fuzz_type, rseed)
        if isfile(join("linetrack_results", f)) and fname in f:
            fuzz_r = pickle.load(open(join("final_policies", f), "rb"))
            pop_summ, pool, tot_warns_mm_e, tot_warns_mm_h, num_cycles, total_trials = fuzz_r
            print(len(pool))
