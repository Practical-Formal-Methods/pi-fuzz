import re
import os
import time
import pickle
import argparse
import EnvWrapper as EW
import Fuzzer
from Oracle import MMSeedBugBasicOracle, MMSeedBugExtOracle, MMSeedBug2BugOracle, MMBugOracle, FailureSeedBugOracle, PerfectSeedBugOracle, PerfectBugOracle, RuleSeedBugOracle
from fuzz_utils import setup_logger

########## box2s-py HAS TO BE INSTALLED!!!!!!!!!!! ##############


def launch_fuzzer(fuzzer):

    logger.info("\n\n")
    logger.info("=" * 30)
    logger.info("Fuzzing starts.")
    logger.info("=" * 30)

    fuzzer.fuzz()

    logger.info("=" * 30)
    logger.info("Fuzzing is over.")
    logger.info("=" * 30)
    
    return [fuzzer.summary, fuzzer.pool, fuzzer.total_trials, fuzzer.schedule.cycles, fuzzer.time_wout_cov]

def test_policy(oracle, pool):
    logger.info("\n")
    logger.info("=" * 30)
    logger.info("Oracle starts testing.")
    logger.info("=" * 30)

    num_tot_bugs = 0
    for idx, fuzz_seed in enumerate(pool):
        num_bugs = oracle.explore(fuzz_seed)
        num_tot_bugs += num_bugs

        fuzz_seed.num_bugs = num_bugs

        logger.info("The oracle has found %d bugs in seed %d" % (num_bugs, idx))

    logger.info("=" * 30)
    logger.info("Oracle finishes testing.")
    logger.info("=" * 30)

    logger.info("Total number of bugs detected by this oracle: %d" % num_tot_bugs)

    return num_bugs


fuzz_start_time = time.strftime("%Y%m%d_%H%M%S")

parser = argparse.ArgumentParser(prog="DebuggingPolicies", description="Find bugs in your policy.")
parser.add_argument("-E", "--env_identifier", choices=['lunar', 'highway', 'bipedal', 'bipedal-hc'], required=True)
parser.add_argument("-R", "--random_seed", type=int, required=True)
parser.add_argument("-A", "--agent_path", required=True)
parser.add_argument("-F", "--fuzz_type", default='inc', choices=['inc', 'non-inc'])
parser.add_argument("-O", "--oracle_type", default="mmseedbugbasic")
parser.add_argument("-C", "--coverage", default="raw", choices=['raw', 'abs'])
parser.add_argument("-CT", "--coverage_thold", default=2.0, type=float)
parser.add_argument("-L", "--logfolder", default="")
parser.add_argument("-FMB", "--fuzz_mut_bdgt", default=25, type=int)  # 25 is OK for lunar and bipedal
parser.add_argument("-IP", "--inf_probability", default=0.1, type=float)

args = parser.parse_args()

env_iden = args.env_identifier
rand_seed = args.random_seed
agent_path = args.agent_path
fuzz_type = args.fuzz_type
oracle_type = args.oracle_type
coverage = args.coverage
coverage_thold = args.coverage_thold
fuzz_mut_bdgt = args.fuzz_mut_bdgt
inf_prob = args.inf_probability
logfolder = args.logfolder
loggername = "fuzz_logger"

if not logfolder:
    logfolder = "pifuzz_logs"
if not os.path.exists(logfolder):
    os.makedirs(logfolder)

logf = "./%s/E%s_R%s_F%s_C%s_I%s_%s.log" % (logfolder, env_iden, rand_seed, fuzz_type, str(coverage_thold), str(inf_prob), fuzz_start_time)

logger = setup_logger(loggername, logf)
logger.info("#############################")
logger.info("### POLICY TESTING REPORT ###")
logger.info("#############################")
logger.info("Policy being tested: %s", agent_path)
logger.info("Environment: %s", env_iden)
logger.info("Random seed: %s", rand_seed)
logger.info("Fuzzer type: %s", fuzz_type)
logger.info("Oracle type: %s", oracle_type)
logger.info("Coverage type: %s", coverage)
logger.info("Coverage thold: %s", coverage_thold)
logger.info("Fuzzer mutation budget: %s", fuzz_mut_bdgt)
logger.info("Informed mutations prob.: %s", inf_prob)

game = EW.Wrapper(env_iden)

# ===============================

fuzz_summ_file = 'E%s_R%s_F%s_C%s_I%s_\d+_\d+_poolonly.p' % (env_iden, rand_seed, fuzz_type, str(coverage_thold), str(inf_prob))

pool_exists = False
for fname in os.listdir(logfolder):
    print(fname)
    if re.search(fuzz_summ_file, fname):
        pool_exists = True  
        fuzz_summ_file = fname      
        break

if not pool_exists:    
    game.create_environment(env_seed=rand_seed)
    game.create_model(agent_path, rand_seed)

    fuzzer = Fuzzer.Fuzzer(rand_seed=rand_seed, fuzz_type=fuzz_type, fuzz_game=game, inf_prob=inf_prob, coverage=coverage, coverage_thold=coverage_thold, mut_budget=fuzz_mut_bdgt)

    fuzz_out = launch_fuzzer(fuzzer)
    pool = fuzzer.pool

    fuzz_summ_file = "./%s/E%s_R%s_F%s_C%s_I%s_%s_poolonly.p" % (logfolder, env_iden, rand_seed, fuzz_type, str(coverage_thold), str(inf_prob), fuzz_start_time)

    fuzzer_summary = open(fuzz_summ_file, "wb")
    pickle.dump(fuzz_out, fuzzer_summary)
else:
    fuzzer_summary = open(logfolder + "/" + fuzz_summ_file, "rb")
    fuzz_out = pickle.load(fuzzer_summary)
    pool = fuzz_out[1]

    logger.info("A pool is already formed with current configurations. Loading that and moving on to the oracle.")

print("Pool size:", len(pool))
    
# ===============================

game.create_environment(env_seed=rand_seed)
game.create_model(agent_path, rand_seed)

oracle_registry = dict()
oracle_registry['mmseedbugbasic'] = MMSeedBugBasicOracle(game, rand_seed)
oracle_registry['mmseedbugext'] = MMSeedBugExtOracle(game, rand_seed)
oracle_registry['mmbug'] = MMBugOracle(game, rand_seed)
oracle_registry['mmseedbug2bug'] = MMSeedBug2BugOracle(game, rand_seed)
oracle_registry['ruleseedbug'] = RuleSeedBugOracle(game, rand_seed)
oracle_registry['failseedbug'] = FailureSeedBugOracle(game, rand_seed)
oracle_registry['perfseedbug'] = PerfectSeedBugOracle(game, rand_seed)
oracle_registry['perfbug'] = PerfectBugOracle(game, rand_seed)

oracle = oracle_registry[oracle_type]

test_start = time.time()
num_bugs = test_policy(oracle, pool)
test_end = time.time()

oracle_time = test_end - test_start
avg_oracle_time = oracle_time / len(pool)

test_out = fuzz_out + [num_bugs, oracle_time, avg_oracle_time]

test_summary = open("./%s/E%s_R%s_O%s_F%s_C%s_I%s_%s.p" % (logfolder, env_iden, rand_seed, oracle_type, fuzz_type, str(coverage_thold), str(inf_prob), fuzz_start_time), "wb")
pickle.dump(test_out, test_summary)

# COMMANDS
# -E highway -R 123 -A policies/linetrack_org.pth -F inc -CT 3.6 -FMB 3 -OMB 2
# -E lunar -R 123 -A policies/lunar_org -F inc -CT 0.6 -FMB 25
# -E bipedal -R 123 -A policies/bipedal_org -F inc -CT 2.0 -FMB 25

