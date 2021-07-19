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

    workbook = xlsxwriter.Workbook('logs/out_%s_%s_dedup_%s.xlsx' % (agent_id, fuzz_start_time, fuzz_type))
    worksheet = workbook.add_worksheet()
    header = ["fuzz_run_id", "bug_type", "coverage", "agent_name", "#easy_warns", "#hard_warns"]
    worksheet.write_row(0, 0, header)

    agent_rngs, env_rngs, fuzz_rngs, orcl_rngs = set_rngs()

    rep_line = 0
    resulting_pools = []
    population_summaries = []
    all_tot_warns_mm_e = []
    all_tot_warns_mm_h = []
    for r_id in range(N_FUZZ_RUNS):

        game = EW.Wrapper()
        game.create_linetrack_environment(rng=env_rngs[r_id])
        mutator = Mutator.RandomActionMutator(game)
        schedule = Scheduler.QueueScheduler()

        logger.info("\n\n")
        logger.info("=" * 30)
        logger.info("Fuzzer started. Fuzz run: %d" % r_id)
        logger.info("=" * 30)

        fuzzer = Fuzzer.Fuzzer(rng=fuzz_rngs[r_id], fuzz_type=fuzz_type, fuzz_game=game, schedule=schedule, mutator=mutator, coverage=coverage)
        pop_summ = fuzzer.fuzz()
        population_summaries.append(pop_summ)
        resulting_pools.append(fuzzer.pool)

        continue

        all_rews = []
        for ap in agent_paths:
            game.create_linetrack_model(load_path=ap, r_seed=r_id)
            rews = []
            for sd in fuzzer.pool:
                env_rng = np.random.default_rng(r_id)
                game.env.reset(rng=env_rng)  # reset the RNG
                game.env.set_state(sd.state_env, sd.data[-1])
                rew, _, fp = game.run_pol_fuzz(sd.data, mode="qualitative")  # this is always qualitative
                rews.append(rew)
            all_rews.append(rews)
        
        mean_rews = np.mean(np.array(all_rews), axis=0)

        fltr_pool = []
        for mr, sd in zip(mean_rews, fuzzer.pool):
            if mr == 100 or mr == 0:
                fltr_pool.append(sd)

        logger.info("Common seeds have been found between %s. Number of common seeds: %d" % (agent_paths, len(fltr_pool)))

        for ap in agent_paths:
            rep_line += 1
            pname = ap.split("/")[-1].split(".")[0]
            logger.info("\n\n")
            logger.info(" *********** Policy %s is being tested.  ***********" % pname)

            orcl_rng = np.random.default_rng(r_id)  # orcl_rngs[r_id]
            mm_oracle = Oracle.MetamorphicOracle(game, mode=bug_type, rng=orcl_rng, de_dup=True)

            game.create_linetrack_model(load_path=ap, r_seed=r_id)

            warnings_mm_e = []
            warnings_mm_h = []
            for idx, fuzz_seed in enumerate(fltr_pool):
                env_rng = np.random.default_rng(r_id)
                game.env.reset(rng=env_rng)   # s[r_id])

                num_warn_mm_e, num_warn_mm_h = mm_oracle.explore(fuzz_seed)
                num_warn_mm_tot = num_warn_mm_e + num_warn_mm_h

                fuzz_seed.num_warn_mm_hard = num_warn_mm_h
                fuzz_seed.num_warn_mm_easy = num_warn_mm_e

                warnings_mm_e.append(num_warn_mm_e)
                warnings_mm_h.append(num_warn_mm_h)

                logger.info("Metamorphic Oracle has found %d(E) + %d(H) = %d warnings in seed %d" % (num_warn_mm_e, num_warn_mm_h, num_warn_mm_tot, idx))

            _, tot_warns_mm_e, _ = post_fuzz_analysis(warnings_mm_e)
            _, tot_warns_mm_h, _ = post_fuzz_analysis(warnings_mm_h)

            all_tot_warns_mm_e.append(tot_warns_mm_e)
            all_tot_warns_mm_h.append(tot_warns_mm_h)

            logger.info("Total number of warnings (E) in this fuzz run: %d" % tot_warns_mm_e)
            logger.info("Total number of warnings (H) in this fuzz run: %d" % tot_warns_mm_h)

            report = [r_id, bug_type, coverage, pname, tot_warns_mm_e, tot_warns_mm_h]
            worksheet.write_row(rep_line, 0, report)

    logger.info("Metamorphic Oracle summary in %d fuzz runs:" % N_FUZZ_RUNS)
    logger.info("    Total number of E warnings: %s" % str(all_tot_warns_mm_e))
    logger.info("    Total number of H warnings: %s" % str(all_tot_warns_mm_h))

    workbook.close()

    return population_summaries, resulting_pools


fuzz_start_time = time.strftime("%Y%m%d_%H%M%S")

# SET SEED IN FUZZ_CONFIG
oracle_type = "metamorphic"
coverage = "raw"
bug_type = "qualitative"
loggername = "fuzz_logger"
logfilename = "logs/policy_testing_%s.log" % fuzz_start_time

parser = argparse.ArgumentParser()
parser.add_argument("agent_name")
parser.add_argument("fuzz_type")
args = parser.parse_args()

agent_id = args.agent_name
fuzz_type = args.fuzz_type

logger = setup_logger(loggername, logfilename)

logger.info("#############################")
logger.info("### POLICY TESTING REPORT ###")
logger.info("#############################")

logger.info("\nRandom Seeds that will be used in %d fuzz runs: %s" % (N_FUZZ_RUNS, RANDOM_SEEDS))
logger.info("Fuzzer type: %s", fuzz_type)
logger.info("Bug Type: %s", bug_type)
logger.info("Coverage Type: %s", coverage)
logger.info("Oracle Type: %s", oracle_type)

ppaths = []
for f in listdir("final_policies"):
    if isfile(join("final_policies", f)) and agent_id in f:
        ppaths.append(join("final_policies", f))

population_summaries_gbox, _ = test_policy(fuzz_type, ppaths, "gbox", coverage)
population_summaries_bbox, _ = test_policy(fuzz_type, ppaths, "bbox", coverage)
plot_rq3_time(population_summaries_gbox, population_summaries_bbox)  # plot graph
