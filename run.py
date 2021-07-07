import time
import argparse
import xlsxwriter
import numpy as np
from os import listdir
from os.path import isfile, join

import Oracle
import Scheduler
import EnvWrapper as EW
import Fuzzer
import Mutator
from fuzz_utils import post_fuzz_analysis, plot_rq3_time, setup_logger, set_rngs
from fuzz_config import RANDOM_SEEDS, N_FUZZ_RUNS


def test_policy(fuzz_type, agent_paths, bug_type, coverage):

    agent_rngs, env_rngs, fuzz_rngs, orcl_rngs = set_rngs()

    resulting_pools = []
    population_summaries = []
    # all_variances_la = []
    all_variances_mm = []
    # all_tot_warns_la = []
    # all_ind_warns_la = []
    all_tot_warns_mm_e = []
    all_tot_warns_mm_h = []
    all_ind_warns_mm = []
    for r_id in range(N_FUZZ_RUNS):

        game = EW.Wrapper()
        game.create_linetrack_environment(rng=env_rngs[r_id])

        mutator = Mutator.RandomActionMutator(game)
        schedule = Scheduler.QueueScheduler()

        fuzzer = Fuzzer.Fuzzer(rng=fuzz_rngs[r_id], fuzz_type=fuzz_type, fuzz_game=game, schedule=schedule, mutator=mutator, coverage=coverage)
        pop_summ = fuzzer.fuzz()
        population_summaries.append(pop_summ)
        resulting_pools.append(fuzzer.pool)

        fltr_pool = []
        for sd in fuzzer.pool:
            rews = []
            for ap in agent_paths:
                game.create_linetrack_model(load_path=ap, r_seed=r_id)
                rew, _, _ = game.run_pol_fuzz(sd.data, mode=bug_type)
                rews.append(rew)
            if np.mean(rews) == 0 or np.mean(rews) == 100:
                fltr_pool.append(sd)

        print(len(fuzzer.pool), len(fltr_pool))
        exit()
        for ap in agent_paths:
            game.create_linetrack_model(load_path=ap, r_seed=r_id)

            orcl_rng = np.random.default_rng(r_id)  # orcl_rngs[r_id]
            mm_oracle = Oracle.MetamorphicOracle(game, mode=bug_type, rng=orcl_rng, de_dup=True)

            logger.info("\n====================")
            logger.info("Fuzz %d Starts Here" % r_id)
            logger.info("====================")
            fuzz_st = time.time()

            warnings_mm_e = []
            warnings_mm_h = []
            for idx, fuzz_seed in enumerate(fltr_pool):
                game.env.set_state(fuzz_seed.state_env, fuzz_seed.data[-1])

                num_warn_mm_e, num_warn_mm_h = mm_oracle.explore(fuzz_seed)
                num_warn_mm_tot = num_warn_mm_e + num_warn_mm_h

                fuzz_seed.num_warn_mm_hard = num_warn_mm_h
                fuzz_seed.num_warn_mm_easy = num_warn_mm_e

                warnings_mm_e.append(num_warn_mm_e)
                warnings_mm_h.append(num_warn_mm_h)

                logger.info("Metamorphic Oracle has found %d(E) + %d(H) = %d warnings in state %d" % (num_warn_mm_e, num_warn_mm_h, num_warn_mm_tot, idx))

            _, tot_warns_mm_e, _ = post_fuzz_analysis(warnings_mm_e)
            _, tot_warns_mm_h, _ = post_fuzz_analysis(warnings_mm_h)

            all_tot_warns_mm_e.append(tot_warns_mm_e)
            all_tot_warns_mm_h.append(tot_warns_mm_h)

            fuzz_et = time.time()

            logger.info("\n=====================================")
            logger.info("Fuzz %d Ends Here. It took %d seconds." % (r_id, int(fuzz_et-fuzz_st)))
            logger.info("======================================")

    # plot_rq3_time(population_summaries, resulting_pools)  # plot graph
    logger.info("Metamorphic Oracle summary in %d fuzz runs:" % N_FUZZ_RUNS)
    logger.info("    Total number of E warnings: %s" % str(all_tot_warns_mm_e))
    logger.info("    Total number of H warnings: %s" % str(all_tot_warns_mm_h))

    return all_tot_warns_mm_e, all_tot_warns_mm_h, all_ind_warns_mm, all_variances_mm  # all_tot_warns_la, all_ind_warns_la, all_variances_la,


fuzz_start_time = time.strftime("%Y%m%d_%H%M%S")

# SET SEED IN FUZZ_CONFIG
oracle_type = "metamorphic"
fuzz_type = "gbox"
coverage = "raw"
bug_type = "qualitative"
loggername = "fuzz_logger"
logfilename = "logs/policy_testing_%s.log" % fuzz_start_time
logger = setup_logger(loggername, logfilename)

logger.info("#############################")
logger.info("### POLICY TESTING REPORT ###")
logger.info("#############################")

logger.info("\nRandom Seeds that will be used in %d fuzz runs: %s" % (N_FUZZ_RUNS, RANDOM_SEEDS))
logger.info("Fuzzer type: %s", fuzz_type)
logger.info("Bug Type: %s", bug_type)
logger.info("Coverage Type: %s", coverage)
logger.info("Oracle Type: %s", oracle_type)


parser = argparse.ArgumentParser()
parser.add_argument("agent_name")
args = parser.parse_args()

agent_id = args.agent_name # "agent8_bad"
ppaths = []
for f in listdir("policies"):
    if isfile(join("policies", f)) and agent_id in f:
        ppaths.append(join("policies", f))

workbook = xlsxwriter.Workbook('logs/out_%s_%s_dedup.xlsx' % (agent_id, fuzz_start_time))
worksheet = workbook.add_worksheet()

tot_mm_e, tot_mm_h, ind_mm, var_mm = test_policy(fuzz_type, ppaths, bug_type, coverage)


for idx, pp in enumerate(ppaths):
    pname = pp.split("/")[-1].split(".")[0]
    logger.info("\n\n**************************************")
    logger.info("Policy %s is being tested." % pname)
    logger.info("**************************************")
    logger.info("Random Action Mutator and Queue Scheduler will be used.")

    tot_mm_e, tot_mm_h, ind_mm, var_mm = test_policy(fuzz_type, pp, bug_type, coverage)

    report = [bug_type, coverage, pname, str(RANDOM_SEEDS), str(tot_mm_e), str(tot_mm_h), str(ind_mm), str(var_mm)]
    worksheet.write_row(idx, 0, report)

    # with open("results/outs.csv", mode="a") as fw:
    #     row = "%s; %s; %s; %s; %s; %s; %s\n" % (bug_type, coverage, pname, RANDOM_SEEDS, tot_mm, ind_mm, var_mm)
    #     fw.write(row)

workbook.close()
