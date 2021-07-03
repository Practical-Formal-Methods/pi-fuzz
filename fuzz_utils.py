import re
import time
import logging
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from fuzz_config import COV_DIST_THOLD, POOL_BUDGET, N_FUZZ_RUNS, RANDOM_SEEDS

def plot_rq3_time(pool_pop_summ, pools):
    for idx, pp in enumerate(pool_pop_summ):
        plt.plot(pp[0], pp[2], lw=2, label=idx+1)
    plt.savefig("results/rq3_poolovertime_timebdgt" + str(POOL_BUDGET) + ".pdf")

    all_warn_seed_times = []
    for pool in pools:
        warn_seed_times = []
        for seed in pool:
            if seed.num_warn_mm_hard or seed.num_warn_mm_easy:
                warn_seed_times.append(seed.fuzz_time)
        all_warn_seed_times.append(warn_seed_times)

    all_warns_over_time = []
    for ws_times in all_warn_seed_times:
        warn_over_time = []
        for sec in range(POOL_BUDGET):
            warn_over_time.append(sum(wst < sec for wst in ws_times))
        all_warns_over_time.append(warn_over_time)

    for wot in all_warns_over_time:
        plt.plot(range(POOL_BUDGET), wot, lw=2)
    plt.savefig("results/rq3_warnovertime_timebdgt" + str(POOL_BUDGET) + ".pdf")


def plot_rq3_trial(pool_pop_summ, pool):
    data_mean = np.array(pool_pop_summ).mean(axis=0)
    data_sigma = np.array(pool_pop_summ).std(axis=0)

    plt.plot(range(POOL_BUDGET), data_mean, lw=2, label='mean', color='blue')
    plt.fill_between(range(POOL_BUDGET), data_mean+data_sigma, data_mean-data_sigma, facecolor='blue', alpha=0.5)
    plt.savefig("results/rq3_pbdgt" + str(POOL_BUDGET) + ".pdf")

def post_fuzz_analysis(warnings):
    var = np.var(warnings)
    ind_warn = 0
    for wrn in warnings:
        if wrn > 0: ind_warn += 1
    ind_warn_norm = 100*ind_warn / len(warnings)
    tot_warn_norm = sum(warnings)

    return ind_warn_norm, tot_warn_norm, var

def set_rngs():
    agent_rngs = []
    fuzz_rngs = []
    orcl_rngs = []
    env_rngs = []
    for i in range(N_FUZZ_RUNS):
        agent_rngs.append(np.random.default_rng(RANDOM_SEEDS[i]))
        env_rngs.append(np.random.default_rng(RANDOM_SEEDS[i]))
        fuzz_rngs.append(np.random.default_rng(RANDOM_SEEDS[i]))
        orcl_rngs.append(np.random.default_rng(RANDOM_SEEDS[i]))

    return agent_rngs, env_rngs, fuzz_rngs, orcl_rngs

def setup_logger(name, log_file, level=logging.DEBUG):
    handler = logging.FileHandler(log_file, mode="w")
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def post_fuzz_reward_analysis(ex_fname, new_fname, game_obj):

    lines_to_keep = []
    with open(ex_fname, 'r') as f:
        cnt = 0
        for line in f:
            level_id = line.split(';')[0]
            game_obj.start_level = int(level_id)
            game_obj.create_environment()
            act_obs_seq = line.split(';')[1].strip()

            act_obs_pairs = re.findall(r'\(.+?\)', act_obs_seq)
            act_seq = [np.array(int(aop[1:-1].split(',')[0])) for aop in act_obs_pairs]

            # seq = [np.array(int(s.strip())) for s in list(seq)]
            # seq = [np.array(int(s.strip())) for s in seq.split(',')]
            org_rew = float(line.split(';')[3].strip())
            _, _, seq_rew, _ = game_obj.run(prefix=act_seq)['data']

            if seq_rew[0] < org_rew: lines_to_keep.append(line)
            cnt += 1
    with open(new_fname, 'w') as f:
        for line in lines_to_keep:  f.write(line)


def post_fuzz_state_analysis(ex_fname, new_fname):
    lines_to_keep = []
    with open(ex_fname, 'r') as f:
        level_id = -1
        observations = set()

        cnt = 0
        for line in f:
            line = line.strip()
            cur_level_id = line.split(';')[0]
            if not cur_level_id == level_id:
                level_id = cur_level_id
                observations = set()

            act_obs_seq = line.split(';')[1]

            act_obs_pairs = re.findall(r'\(.+?\)', act_obs_seq)
            last_pair = act_obs_pairs[-1]
            last_obs = re.findall(r'\[.+?\]', last_pair)[0]

            if last_obs not in observations:
                observations.add(last_obs)
                lines_to_keep.append(line)

            cnt += 1
            print(cnt, len(lines_to_keep))

    with open(new_fname, 'w') as f:
        for line in lines_to_keep:  f.write(line + '\n')

# LEGACY
# def check_safe_state(game, path, action_space=[0,1,2,3,4]):
#     lookahead_rewards = []
#
#     dev_paths = []
#     # for repeat in range(LOOKAHEAD_DEPTH):
#     dev_paths = list(itertools.product(action_space, repeat=LOOKAHEAD_DEPTH))
#
#     for dp in dev_paths:
#         res = game.run_pol_fuzz(prefix=path+list(dp), safe_check=True)
#         _, rew, _, _ = res["data"]
#         lookahead_rewards.append(rew)
#
#     return lookahead_rewards

def state_abstraction(cand_states):
    l2_distance_thold = 2
    for cand in cand_states:
        pass
    return True


def plot_tims_vs_warnings(fname):
    run_identifier = 'linetrack'
    secs = []
    wrngs = []
    with open(fname, 'r') as fr:
        for line in fr:
            line = line.strip()
            sec = float(line.split(',')[0].strip())
            wrng = float(line.split(',')[1].strip())
            secs.append(sec)
            wrngs.append(wrng)

    secs = [s-run_identifier for s in secs]  # file_identifier is the time the fuzzer begun

    plt.plot(secs, wrngs)
    plt.ylabel('num warnings')
    plt.xlabel('time')
    plt.savefig('time_vs_wrn.png')


def log_time_vs_warnings(env_name, num_warnings):
    run_identifier = 'linetrack'
    cur_time = time.time()
    with open('logs/' + env_name + '_time_vs_warnings' + run_identifier.replace(' ', '_') + '.txt', 'a') as fw:
        fw.write(str(cur_time) + ', ' + str(num_warnings) + '\n')


def emit_warning(env_name, start_level, org_reward, policy, new_reward):
    run_identifier = 'linetrack'
    # print("Overfitting state found: ", policy)
    # act_obs_pairs = []
    # for idx, pc in enumerate(pre_cov):
    #     act_obs_pairs.append((policy[idx][0], list(pc)))

    with open('logs/' + env_name + 'warnings' + run_identifier.replace(' ', '_') + '.txt', 'a') as fw:
        fw.write(str(start_level) + '; ' + str(policy) + '; ' + str(new_reward[0]) + '; ' + str(
            org_reward[0]))
        fw.write('\n')


def mean_confidence_interval(data, confidence=0.99):
    a = 1.0 * np.array(data)
    m, se = np.mean(a), scipy.stats.sem(a)

    return scipy.stats.norm.interval(confidence, loc=m, scale=se)


def time_bound_exceeded(fuzzer_start_time, bound_in_hours=12):  # todo hyperparamter
    current_time = time.time()
    if current_time - fuzzer_start_time > bound_in_hours * 60 * 60:
        return True

    return False

def calculate_entropy(env_name, action_probs):

    if env_name == 'maze':
        eq_action_probs = []
        for ap in action_probs:
            ap = list(ap)
            new_ap = []
            new_ap.append(sum(ap[:3]))
            new_ap.append(ap[3])
            new_ap.append(ap[4] + sum(ap[9:]))
            new_ap.append(ap[5])
            new_ap.append(sum(ap[6:9]))

            eq_action_probs.append(new_ap)
    else:
        eq_action_probs = action_probs

    entropies = []
    for ap in eq_action_probs: entropies.append(entropy(ap))

    entropies = [float(e) for e in entropies]

    return entropies


def plot_entropy(entropies):
    plt.plot(range(1, len(entropies)+1), entropies, marker='o')
    plt.show()


# LEGACY
# def mutate_seed_selection(scheduler, pool, org_pol_len, explored_seeds):
#     seed = scheduler.choose(pool)
#     new_seed_data = seed.data
#     trial = 0
#     while new_seed_data in explored_seeds:
#         new_seed_data += random.normalvariate(MUTATE_MU, MUTATE_SIGMA)
#         if new_seed_data < 0: new_seed_data = -new_seed_data
#         if new_seed_data > (org_pol_len - 1): new_seed_data = org_pol_len - 1
#         new_seed_data = int(new_seed_data)
#         trial += 1
#         if trial >= 10: random_seed_selection(org_pol_len, explored_seeds)
#
#     return seed, new_seed_data


def entropy_seed_selection(org_pol_len, entropies, explored_seeds):
    choice_list = [d for d in range(org_pol_len) if d not in explored_seeds]
    choice_entr = [entropies[d] for d in choice_list]
    rec_entropies = np.reciprocal(choice_entr)
    norm_rec_entropies = rec_entropies / np.sum(rec_entropies)
    new_seed_data = np.random.choice(choice_list, p=norm_rec_entropies)

    return new_seed_data


def random_seed_selection(org_pol_len, explored_seeds):
    choice_list = [d for d in range(org_pol_len) if d not in explored_seeds]
    new_seed_data = np.random.choice(choice_list)

    return new_seed_data

# @profile
def coverage_update(covered, queries):
    new_coverage = False
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric='euclidean').fit(covered)
    for query in queries:
        distance, _ = nbrs.kneighbors([query])
        if distance[0] > COV_DIST_THOLD:
            covered.append(query)
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric='euclidean').fit(covered)
            new_coverage = True

    return covered, new_coverage