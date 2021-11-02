import time
import pickle
import argparse
import Oracle
import EnvWrapper as EW
import Fuzzer
from fuzz_utils import post_fuzz_analysis, setup_logger


def test_policy(env_identifier, fuzz_type, agent_path, bug_type, coverage, coverage_thold, r_seed, fuzz_mut_bdgt, orcl_mut_bdgt, inf_prob, delta):

    game = EW.Wrapper(env_identifier)
    game.create_environment(env_seed=r_seed)
    game.create_model(agent_path, r_seed)

    logger.info("\n\n")
    logger.info("=" * 30)
    logger.info("Fuzzing starts.")
    logger.info("=" * 30)

    fuzzer = Fuzzer.Fuzzer(r_seed=r_seed, fuzz_type=fuzz_type, fuzz_game=game, inf_prob=inf_prob, coverage=coverage, coverage_thold=coverage_thold, mut_budget=fuzz_mut_bdgt)
    pop_summ = fuzzer.fuzz()

    print("Pool size:", len(fuzzer.pool))
    pickle.dump([pop_summ, fuzzer.pool, fuzzer.total_trials], open("%s_%s_%d_%s_sp%f_poolonly.p"%(env_identifier, fuzz_type, r_seed, fuzz_start_time, inf_prob), "wb"))
    
    rep_line = 0
    rep_line += 1
    pname = agent_path.split("/")[-1].split(".")[0]
    logger.info("\n\n")
    logger.info(" *********** Policy %s is starting to be tested. ***********" % pname)

    game.create_environment(env_seed=r_seed)
    game.create_model(agent_path, r_seed)
    mm_oracle = Oracle.MetamorphicOracle(game, mode=bug_type, r_seed=r_seed, delta=delta, orcl_mut_bdgt=orcl_mut_bdgt, de_dup=True)

    total_time = 0
    tot_num_rejects = 0
    warnings_mm_e = []
    warnings_mm_h = []
    for idx, fuzz_seed in enumerate(fuzzer.pool):
        s = time.time()
        num_warn_mm_e, num_warn_mm_h, num_rejects = mm_oracle.explore(fuzz_seed)
        e = time.time()
        total_time += (e-s)
        num_warn_mm_tot = num_warn_mm_e + num_warn_mm_h

        fuzz_seed.num_warn_mm_hard = num_warn_mm_h
        fuzz_seed.num_warn_mm_easy = num_warn_mm_e

        warnings_mm_e.append(num_warn_mm_e)
        warnings_mm_h.append(num_warn_mm_h)

        tot_num_rejects += num_rejects
        logger.info("Metamorphic Oracle has found %d(E) + %d(H) = %d warnings in seed %d. Num rejects: %d." % (num_warn_mm_e, num_warn_mm_h, num_warn_mm_tot, idx, num_rejects))

    avg_time = total_time / len(fuzzer.pool)
    _, tot_warns_mm_e, _ = post_fuzz_analysis(warnings_mm_e)
    _, tot_warns_mm_h, _ = post_fuzz_analysis(warnings_mm_h)

    logger.info("Total number of warnings (E) in this fuzz run: %d" % tot_warns_mm_e)
    logger.info("Total number of warnings (H) in this fuzz run: %d" % tot_warns_mm_h)
    logger.info("Total number of rejected Oracle mutations in this fuzz run: %d" % tot_num_rejects)

    logger.info("Metamorphic Oracle summary:")
    logger.info("    Total number of E warnings: %s" % str(tot_warns_mm_e))
    logger.info("    Total number of H warnings: %s" % str(tot_warns_mm_h))

    num_cycles = fuzzer.schedule.cycles
    total_trials = fuzzer.total_trials

    print(len(pop_summ), len(fuzzer.pool), tot_warns_mm_e, tot_warns_mm_h, num_cycles, total_trials, total_time, avg_time)
    return pop_summ, fuzzer.pool, tot_warns_mm_e, tot_warns_mm_h, num_cycles, total_trials, total_time, avg_time


fuzz_start_time = time.strftime("%Y%m%d_%H%M%S")

parser = argparse.ArgumentParser(prog="DebuggingPolicies", description="Find bugs in your policy.")
parser.add_argument("-E", "--env_identifier", choices=['lunar', 'linetrack', 'bipedal', 'bipedal-hc', 'racetrack'], required=True)
parser.add_argument("-R", "--random_seed", type=int, required=True)
parser.add_argument("-A", "--agent_path", required=True)
parser.add_argument("-F", "--fuzz_type", default='inc', choices=['inc', 'non-inc'])
parser.add_argument("-O", "--oracle_type", default="metamorphic")
parser.add_argument("-B", "--bug_type", default="qualitative", choices=['qualitative', 'quantitative'])
parser.add_argument("-C", "--coverage", default="raw", choices=['raw', 'abs'])
parser.add_argument("-CT", "--coverage_thold", default=2.0, type=float)
parser.add_argument("-L", "--logfile", default="logs/policy_testing_%s.log" % fuzz_start_time)
parser.add_argument("-FMB", "--fuzz_mut_bdgt", default=25, type=int)  # 25 is OK for lunar and bipedal
parser.add_argument("-OMB", "--orcl_mut_bdgt", default=25, type=int)
parser.add_argument("-D", "--delta", default=1.0, type=float)
parser.add_argument("-IP", "--inf_probability", default=0.1, type=float)

args = parser.parse_args()

env_iden = args.env_identifier
rand_seed = args.random_seed
agent_path = args.agent_path
fuzz_type = args.fuzz_type
oracle_type = args.oracle_type
bug_type = args.bug_type
coverage = args.coverage
coverage_thold = args.coverage_thold
fuzz_mut_bdgt = args.fuzz_mut_bdgt
orcl_mut_bdgt = args.orcl_mut_bdgt
logfilename = args.logfile
delta = args.delta
inf_prob = args.inf_probability

loggername = "fuzz_logger"

logger = setup_logger(loggername, logfilename)
logger.info("#############################")
logger.info("### POLICY TESTING REPORT ###")
logger.info("#############################")
logger.info("Policy being tested: %s", agent_path)
logger.info("Fuzzer type: %s", fuzz_type)
logger.info("Bug Type: %s", bug_type)
logger.info("Coverage Type: %s", coverage)
logger.info("Oracle Type: %s", oracle_type)

test_out = test_policy(env_iden, fuzz_type, agent_path, bug_type, coverage, coverage_thold, rand_seed, fuzz_mut_bdgt, orcl_mut_bdgt, inf_prob, delta)
pickle.dump(test_out, open("%s_%s_%d_%s_sp%f.p" % (env_iden, fuzz_type, rand_seed, fuzz_start_time, inf_prob), "wb"))

# COMMANDS
# -E linetrack -R 123 -A policies/linetrack_org.pth -F inc -CT 3.6 -FMB 3
# -E lunar -R 123 -A policies/lunar_org -F inc -CT 0.6 -FMB 25
# -E bipedal -R 123 -A policies/bipedal_org -F inc -CT 2.0 -FMB 25
# -E racetrack -R 123 -A policies/racetrack_org -F inc -CT  -FMB

