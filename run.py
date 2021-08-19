import time
import pickle
import argparse
import xlsxwriter
import Oracle
import EnvWrapper as EW
import Fuzzer
from fuzz_utils import post_fuzz_analysis, setup_logger


def test_policy(env_identifier, fuzz_type, agent_path, bug_type, coverage, coverage_thold, r_seed, fuzz_mut_bdgt, orcl_mut_bdgt, use_seedp, delta):

    workbook = xlsxwriter.Workbook('logs_/out_%s_dedup_%s.xlsx' % (fuzz_start_time, fuzz_type))
    worksheet = workbook.add_worksheet()
    header = ["random_seed", "bug_type", "coverage", "agent_name", "#easy_warns", "#hard_warns"]
    worksheet.write_row(0, 0, header)

    game = EW.Wrapper(env_identifier)
    game.create_environment(env_seed=r_seed)
    game.create_model(agent_path, r_seed)

    logger.info("\n\n")
    logger.info("=" * 30)
    logger.info("Fuzzing starts.")
    logger.info("=" * 30)

    fuzzer = Fuzzer.Fuzzer(r_seed=r_seed, fuzz_type=fuzz_type, fuzz_game=game, use_seedp=use_seedp, coverage=coverage, coverage_thold=coverage_thold, mut_budget=fuzz_mut_bdgt)
    pop_summ = fuzzer.fuzz()

    #pickle.dump([fuzzer.pool, pop_summ], open("%s_%s_%d_%s_nosp_poolonly.p"%(env_identifier, fuzz_type, r_seed, fuzz_start_time), "wb"))
    print("Pool size nosp:", len(fuzzer.pool))
    
    rep_line = 0
    # for ap in agent_paths:
    rep_line += 1
    pname = agent_path.split("/")[-1].split(".")[0]
    logger.info("\n\n")
    logger.info(" *********** Policy %s is starting to be tested. ***********" % pname)

    game.create_environment(env_seed=r_seed)
    game.create_model(agent_path, r_seed)
    mm_oracle = Oracle.MetamorphicOracle(game, mode=bug_type, r_seed=r_seed, delta=delta, orcl_mut_bdgt=orcl_mut_bdgt, de_dup=True)

    # game.create_linetrack_model(load_path=ap, r_seed=r_id)

    tot_num_rejects = 0
    warnings_mm_e = []
    warnings_mm_h = []
    for idx, fuzz_seed in enumerate(fuzzer.pool):
        num_warn_mm_e, num_warn_mm_h, num_rejects = mm_oracle.explore(fuzz_seed)
        num_warn_mm_tot = num_warn_mm_e + num_warn_mm_h

        fuzz_seed.num_warn_mm_hard = num_warn_mm_h
        fuzz_seed.num_warn_mm_easy = num_warn_mm_e

        warnings_mm_e.append(num_warn_mm_e)
        warnings_mm_h.append(num_warn_mm_h)

        tot_num_rejects += num_rejects
        logger.info("Metamorphic Oracle has found %d(E) + %d(H) = %d warnings in seed %d. Num rejects: %d." % (num_warn_mm_e, num_warn_mm_h, num_warn_mm_tot, idx, num_rejects))

    _, tot_warns_mm_e, _ = post_fuzz_analysis(warnings_mm_e)
    _, tot_warns_mm_h, _ = post_fuzz_analysis(warnings_mm_h)

    logger.info("Total number of warnings (E) in this fuzz run: %d" % tot_warns_mm_e)
    logger.info("Total number of warnings (H) in this fuzz run: %d" % tot_warns_mm_h)
    logger.info("Total number of rejected Oracle mutations in this fuzz run: %d" % tot_num_rejects)

    report = [r_seed, bug_type, coverage, pname, tot_warns_mm_e, tot_warns_mm_h]
    worksheet.write_row(rep_line, 0, report)

    logger.info("Metamorphic Oracle summary:")
    logger.info("    Total number of E warnings: %s" % str(tot_warns_mm_e))
    logger.info("    Total number of H warnings: %s" % str(tot_warns_mm_h))

    workbook.close()

    num_cycles = fuzzer.schedule.cycles
    total_trials = fuzzer.total_trials

    print(len(pop_summ), len(fuzzer.pool), tot_warns_mm_e, tot_warns_mm_h, num_cycles, total_trial)
    return pop_summ, fuzzer.pool, tot_warns_mm_e, tot_warns_mm_h, num_cycles, total_trials


fuzz_start_time = time.strftime("%Y%m%d_%H%M%S")

parser = argparse.ArgumentParser(prog="DebuggingPolicies", description="Find bugs in your policy.")
parser.add_argument("-E", "--env_identifier", choices=['lunar', 'linetrack', 'bipedal', 'bipedal-hc'], required=True)
parser.add_argument("-R", "--random_seed", type=int, required=True)
parser.add_argument("-A", "--agent_path", required=True)
parser.add_argument("-F", "--fuzz_type", default='gbox', choices=['gbox', 'bbox'])
parser.add_argument("-O", "--oracle_type", default="metamorphic")
parser.add_argument("-B", "--bug_type", default="qualitative", choices=['qualitative', 'quantitative'])
parser.add_argument("-C", "--coverage", default="raw", choices=['raw', 'abs'])
parser.add_argument("-CT", "--coverage_thold", default=2.0, type=float)  # 0.75 for lunar, 2.0 for bipedal
parser.add_argument("-L", "--logfile", default="logs_/policy_testing_%s.log" % fuzz_start_time)
parser.add_argument("-FMB", "--fuzz_mut_bdgt", default=25, type=int)  # 25 is OK for lunar and ipedal
parser.add_argument("-OMB", "--orcl_mut_bdgt", default=25, type=int)
parser.add_argument("-D", "--delta", default=1.0, type=float)
parser.add_argument("-USP", "--use_sp", action='store_true', default=False)

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
use_seedp = args.use_sp

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

test_out = test_policy(env_iden, fuzz_type, agent_path, bug_type, coverage, coverage_thold, rand_seed, fuzz_mut_bdgt, orcl_mut_bdgt, use_seedp, delta)
pickle.dump(test_out, open("%s_%s_%d_%s.p"%(env_iden, fuzz_type, rand_seed, fuzz_start_time), "wb"))

# -E linetrack -R 123 -A final_policies/linetrack_org.pth -F gbox -CT 5.0 -FMB 3 -OMB 1
# -E lunar -R 123 -A final_policies/lunar_org -F gbox -CT 0.75 -FMB 25 -OMB 25


#
# obs = game.env.reset()
# acts = []
# for i in range(120):
#     action = agent_rngs[0].choice(range(game.env.action_space.n), 1)[0]
#     game.env.render()
#     time.sleep(0.1)
#     _, s = game.get_state()
#     if i == 30:
#         ss = s  #copy.deepcopy(s)
#         time.sleep(3)
#     if i >= 30 and i < 35:
#         print(action, s)
#         acts.append(action)
#     obs, rewards, dones, info = game.env.step(action)
#
# game.create_environment(env_seed=RANDOM_SEEDS[r_id])
# print("here")
# game.set_state(hi_lvl_state=ss)
# for i in range(5):
#     # action = agent_rngs[0].choice(range(game.env.action_space.n), 1)[0]
#     action = acts[i]
#     game.env.render()
#     _, s = game.get_state()
#     print(action, s)
#     obs, rewards, dones, info = game.env.step(action)
#
#     if i == 0:
#         time.sleep(3)
#     else:
#         time.sleep(0.1)
# exit()

    #
    # obsv_dim, action_dim, continuous_action_space = get_env_space()
    # actor = Actor(obsv_dim,
    #               action_dim,
    #               continuous_action_space=continuous_action_space,
    #               trainable_std_dev=hp.trainable_std_dev,
    #               init_log_std_dev=hp.init_log_std_dev)
    # critic = Critic(obsv_dim)
    #
    # actor_state_dict, critic_state_dict, actor_optimizer_state_dict, critic_optimizer_state_dict, _ = load_checkpoint(60)
    #
    # actor.load_state_dict(actor_state_dict, strict=True)
    # critic.load_state_dict(critic_state_dict, strict=True)
    #
    # # game.create_model("actor.pth")  # final_policies/bipedal_easy_900K")  # final_policies/ppo_bipedal_400000")
    # obs = game.env.reset()
    # print(obs.dtype)
    # obs = list(obs)
    # print(obs)
    # import torch
    # tens = torch.DoubleTensor(obs)
    # print(tens.dtype)
    # exit()
    # tens = torch.from_numpy(np.expand_dims(np.expand_dims(obs, axis=0), axis=0))
    # # tens = torch.from_numpy(obs)
    # print(tens.dtype)
    # tens.double()
    # actor.forward(tens.double())
    # exit()
    # print(tens)
    # tens = tens.type(torch.DoubleTensor)
    # print(tens)
    # tens.to(torch.long)
    # print(tens)
    # exit()
    # rew, fp = game.run_pol_fuzz(obs, mode="quantitative", render=True)  # this is always qualitative
    # print(rew)
    # exit()


    ## LEGACY CODE
    # all_rews = []
    # for ap in agent_paths:
    #     game.create_model(ap)
    #     rews = []
    #     for sd in fuzzer.pool:
    #         game.env.seed(r_seed)
    #         game.set_state(sd.hi_lvl_state)
    #         rew, fp = game.run_pol_fuzz(sd.data, mode="qualitative", render=False)  # this is always qualitative
    #         rews.append(rew)
    #     all_rews.append(rews)
    # mean_rews = np.mean(np.array(all_rews), axis=0)
    #
    # fltr_pool = []
    # for mr, sd in zip(mean_rews, fuzzer.pool):
    #     if mr == 100 or mr == 0:
    #         fltr_pool.append(sd)
    # logger.info("Common seeds have been found between %s.\nNumber of common seeds: %d" % (agent_paths, len(fltr_pool)))
