import time
from os import listdir
from os.path import isfile, join

import Oracle
import Scheduler
import EnvWrapper as EW
import Fuzzer
import Mutator
from fuzz_utils import post_fuzz_analysis, plot_rq3, setup_logger
from fuzz_config import RANDOM_SEED

def fuzz_func(agent_path, bug_type, coverage):

    game = EW.Wrapper(agent_path)
    game.create_linetrack_environment()
    logger.info("Environment created with following parameters:  mode=line_ratio, input_stripe=True, num_lines = 2, length_lines = 100, ratios = [0.02, 0.1]")
    game.create_linetrack_model()
    logger.info("Policy NN loaded from " + agent_path)

    mutator = Mutator.RandomActionMutator(game)
    # schedule = Scheduler.RandomScheduler()  # using random scheduler can be problematic. check pool population code
    schedule = Scheduler.QueueScheduler()
    logger.info("Random Action Mutator and Queue Scheduler are used.")
    la_oracle = Oracle.LookAheadOracle(game, mode=bug_type)
    mm_oracle = Oracle.MetamorphicOracle(game, mode=bug_type)

    population_summaries = []
    all_variances_la = []
    all_variances_mm = []
    all_tot_warns_la = []
    all_ind_warns_la = []
    all_tot_warns_mm = []
    all_ind_warns_mm = []
    fuzz_runs = 8
    for r_id in range(fuzz_runs):
        logger.info("\n====================")
        logger.info("Fuzz %d Starts Here" % r_id)
        logger.info("====================")
        fuzz_st = time.time()

        fuzzer = Fuzzer.Fuzzer(fuzz_game=game, schedule=schedule, la_oracle=la_oracle, mm_oracle=mm_oracle, mutator=mutator, coverage=coverage)
        warnings_la, warnings_mm, pop_summ = fuzzer.fuzz()
        population_summaries.append(pop_summ)

        ind_warns_la, tot_warns_la, var_la = post_fuzz_analysis(warnings_la)
        ind_warns_mm, tot_warns_mm, var_mm = post_fuzz_analysis(warnings_mm)

        logger.info("\nLookahead Oracle has found total of %d warnings on %.2f%% percent of states (%d). Variance is: %.2f" % (tot_warns_la, ind_warns_la, len(fuzzer.pool), var_la))
        logger.info("Metamoprhic Oracle has found total of %d warnings on %.2f%% percent of states (%d). Variance is: %.2f" % (tot_warns_mm, ind_warns_mm, len(fuzzer.pool), var_mm))

        all_variances_la.append(var_la)
        all_variances_mm.append(var_mm)
        all_ind_warns_la.append(ind_warns_la)
        all_ind_warns_mm.append(ind_warns_mm)
        all_tot_warns_la.append(tot_warns_la)
        all_tot_warns_mm.append(tot_warns_mm)

        fuzz_et = time.time()

        logger.info("\n=====================================")
        logger.info("Fuzz %d Ends Here. It took %d seconds." % (r_id, int(fuzz_et-fuzz_st)))
        logger.info("======================================")

    logger.info("Lookahead Oracle summary in %d fuzz runs:" % fuzz_runs)
    logger.info("    Total number of warnings: %s" % str(all_tot_warns_la))
    logger.info("    Variance in number of warnings found in states: %s" % str(all_variances_la))
    logger.info("    Percentage of states with at least one warning: %s" % str(all_ind_warns_la))
    logger.info("Metamorphic Oracle summary in %d fuzz runs:" % fuzz_runs)
    logger.info("    Total number of warnings: %s" % str(all_tot_warns_mm))
    logger.info("    Variance in number of warnings found in states: %s" % str(all_variances_mm))
    logger.info("    Percentage of states with at least one warning: %s" % str(all_ind_warns_mm))

    plot_rq3(population_summaries)

    return all_tot_warns_la, all_ind_warns_la, all_variances_la, all_tot_warns_mm, all_ind_warns_mm, all_variances_mm

# SET SEED IN FUZZ_CONFIG
# oracle_type = "metamorphic"
coverage = "raw"
bug_type = "qualitative"
loggername = "fuzz_logger"
logfilename = "policy_testing_%s.log" % time.strftime("%Y%m%d_%H%M%S")
logger = setup_logger(loggername, logfilename)

logger.info("#############################")
logger.info("### POLICY TESTING REPORT ###")
logger.info("#############################")

logger.info("\nRandom Seed: %d", RANDOM_SEED)
logger.info("Bug Type: %d", bug_type)
logger.info("Coverage Type: %d", coverage)

ppaths = []
for f in listdir("policies"):
    if isfile(join("policies", f)):
        ppaths.append(join("policies", f))


for idx, pp in enumerate(ppaths):
    pname = pp.split("/")[-1].split(".")[0]
    logger.info("\n\n==================================")
    logger.info("==================================")
    logger.info("Policy %s is being tested." % pname)
    logger.info("==================================")
    logger.info("==================================")

    tot_la, ind_la, var_la, tot_mm, ind_mm, var_mm = fuzz_func(pp, bug_type, coverage)

    with open("outs.csv", mode="a") as fw:
        row = "%s, %s, %s, %d, %s, %s, %s, %s, %s, %s\n" % (bug_type, coverage, pname, RANDOM_SEED, tot_la, ind_la, var_la, tot_mm, ind_mm, var_mm)
        fw.write(row)

