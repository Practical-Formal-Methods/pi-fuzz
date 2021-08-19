import re
import time
import logging
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from fuzz_config import FUZZ_BUDGET

def plot_rq3_time_cloud(pool_pop_summ_gb, pool_pop_summ_bb):

    all_size_gb = []
    for pp in pool_pop_summ_gb:
        print("yep")
        size_gb = []
        pp = np.array(pp)
        times = pp[:, 1]
        lptr = 0
        for sec in range(FUZZ_BUDGET)[0::100]:
            cnt = 0
            for tm in times[lptr:]:
                if tm > sec:
                    break
                cnt += 1
            if cnt == 0:
                size_gb.append(0)
            else:
                size_gb.append(pp[:, 2][cnt-1])

            lptr = cnt
        all_size_gb.append(size_gb)

    all_size_bb = []
    for pp in pool_pop_summ_bb:
        print("yep2")
        size_bb = []
        pp = np.array(pp)
        times = pp[:, 1]
        lptr = 0
        for sec in range(FUZZ_BUDGET)[0::100]:
            cnt = 0
            for tm in times[lptr:]:
                if tm > sec:
                    break
                cnt += 1
            if cnt == 0:
                size_bb.append(0)
            else:
                size_bb.append(pp[:, 2][cnt-1])

            lptr = cnt
        all_size_bb.append(size_bb)


    all_size_gb_mean = np.array(all_size_gb).mean(axis=0)
    all_size_bb_mean = np.array(all_size_bb).mean(axis=0)

    all_size_gb_std = np.array(all_size_gb).std(axis=0)
    all_size_bb_std = np.array(all_size_bb).std(axis=0)


    plt.plot(range(FUZZ_BUDGET)[0::100], all_size_gb_mean, lw=2, label='graybox', color='blue')
    plt.fill_between(range(FUZZ_BUDGET)[0::100], all_size_gb_mean+all_size_gb_std, all_size_gb_mean-all_size_gb_std, facecolor='blue', alpha=0.5)

    plt.plot(range(FUZZ_BUDGET)[0::100], all_size_bb_mean, lw=2, label='blackbox', color='red')
    plt.fill_between(range(FUZZ_BUDGET)[0::100], all_size_bb_mean+all_size_bb_std, all_size_bb_mean-all_size_bb_std, facecolor='red', alpha=0.5)

    plt.xlabel("Time (sec)")
    plt.ylabel("Pool Size")
    plt.legend(loc="upper left")

    plt.savefig("results/rq3_poolsize_overtime_cloud_timebdgt_%d.pdf" % (FUZZ_BUDGET) )

def plot_rq3_time(pool_pop_summ_gb, pool_pop_summ_bb):

    for pp in pool_pop_summ_gb:
        pp = np.array(pp)
        plt.plot(pp[:, 1], pp[:, 2], lw=2)

    for pp in pool_pop_summ_bb:
        pp = np.array(pp)
        plt.plot(pp[:, 1], pp[:, 2], "--", lw=2)

    plt.savefig("results/rq3_poolovertime_timebdgt" + str(FUZZ_BUDGET) + ".pdf")


def sub_rq3_warn(pools):
    all_warn_seed_times = []
    for pool in pools:
        warn_seed_times = []
        for seed in pool:
            if seed.num_warn_mm_hard or seed.num_warn_mm_easy:
                warn_seed_times.append(seed.gen_time)
        all_warn_seed_times.append(warn_seed_times)

    all_warns_over_time = []
    for ws_times in all_warn_seed_times:
        warn_over_time = []
        for sec in range(FUZZ_BUDGET):
            warn_over_time.append(sum(wst < sec for wst in ws_times))
        all_warns_over_time.append(warn_over_time)
    
    return all_warns_over_time

def plot_rq3_warn(agent_id, pools_g, pools_b):
    
    all_warns_over_time_g = sub_rq3_warn(pools_g)
    all_warns_over_time_b = sub_rq3_warn(pools_b)

    a_w_o_t_g_mean = np.array(all_warns_over_time_g).mean(axis=0)
    a_w_o_t_b_mean = np.array(all_warns_over_time_b).mean(axis=0)

    a_w_o_t_g_std = np.array(all_warns_over_time_g).std(axis=0)
    a_w_o_t_b_std = np.array(all_warns_over_time_b).std(axis=0)
    
    '''
    for wot in all_warns_over_time_g:
        plt.plot(range(POOL_BUDGET), wot, lw=2)
    
    for wot in all_warns_over_time_b:
        plt.plot(range(POOL_BUDGET), wot, "--", lw=2)
    '''

    plt.plot(range(FUZZ_BUDGET), a_w_o_t_g_mean, lw=2, label='graybox', color='blue')
    plt.fill_between(range(FUZZ_BUDGET), a_w_o_t_g_mean+a_w_o_t_g_std, a_w_o_t_g_mean-a_w_o_t_g_std, facecolor='blue', alpha=0.5)
    
    plt.plot(range(FUZZ_BUDGET), a_w_o_t_b_mean, lw=2, label='blackbox', color='red')
    plt.fill_between(range(FUZZ_BUDGET), a_w_o_t_b_mean+a_w_o_t_b_std, a_w_o_t_b_mean-a_w_o_t_b_std, facecolor='red', alpha=0.5)

    plt.xlabel("Time(sec)")
    plt.ylabel("# Warnings")
    plt.legend(loc="upper left")

    plt.savefig("results/rq3_warnovertime_mustd_timebdgt_%d_%s.pdf" % (FUZZ_BUDGET, agent_id) )


def plot_rq3_trial(pool_pop_summ, pool):
    data_mean = np.array(pool_pop_summ).mean(axis=0)
    data_sigma = np.array(pool_pop_summ).std(axis=0)

    plt.plot(range(FUZZ_BUDGET), data_mean, lw=2, label='mean', color='blue')
    plt.fill_between(range(FUZZ_BUDGET), data_mean+data_sigma, data_mean-data_sigma, facecolor='blue', alpha=0.5)
    plt.savefig("results/rq3_pbdgt" + str(FUZZ_BUDGET) + ".pdf")

def post_fuzz_analysis(warnings):
    var = np.var(warnings)
    ind_warn = 0
    for wrn in warnings:
        if wrn > 0: ind_warn += 1
    ind_warn_norm = 100*ind_warn / len(warnings)
    tot_warn_norm = sum(warnings)

    return ind_warn_norm, tot_warn_norm, var

# LEGACY CODE
def set_rngs():
    agent_rngs = []
    fuzz_rngs = []
    orcl_rngs = []
    for i in range(N_FUZZ_RUNS):
        agent_rngs.append(np.random.default_rng(RANDOM_SEEDS[i]))
        fuzz_rngs.append(np.random.default_rng(RANDOM_SEEDS[i]))
        orcl_rngs.append(np.random.default_rng(RANDOM_SEEDS[i]))

    return fuzz_rngs, orcl_rngs

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

def read_outs_excel(folder=None, fuzz_type="gbox"):
    if folder is None:
        folder = "outs"

    all_warns = [[], [], [], []]
    for f in listdir(folder):
        if isfile(join(folder, f)) and fuzz_type in f:
            agent_id = f.split("_")[1]
            print(agent_id)
            ppath = join(folder, f)
            data = pd.read_excel(ppath, engine='openpyxl')
            df = pd.DataFrame(data, columns=['agent_name', 'bug_type', '#easy_warns', '#hard_warns'])
            bug_type = df['bug_type'].unique()[0]
            agent_names = df['agent_name'].unique()
            agnt_ord_id = []
            for ag_nm in agent_names:
                agnt_ord_id.append(int(ag_nm.split('_')[1]))
            agent_names_sorted = [ag_nm for _, ag_nm in sorted(zip(agnt_ord_id, agent_names), key=lambda pair: pair[0])]
            for idx, ag_nm in enumerate(agent_names_sorted):
                sub_df = df.loc[df['agent_name'] == ag_nm]
                warns = sub_df["#easy_warns"].to_numpy()
                all_warns[idx] += list(warns)

    boxplot("all_agents", fuzz_type, bug_type, all_warns)

def boxplot(env_idn, fuzz_types, num_tot_warn):
    green_diamond = dict(markerfacecolor="g", marker="D")
    fig, ax = plt.subplots()
    ax.set_ylabel("# Warnings")
    ax.set_xlabel("Agent Quality (Higher Better)")
    ax.boxplot(num_tot_warn, flierprops=green_diamond)
    ax.set_xticklabels(fuzz_types)
    plt.savefig("num_warn_boxplot_%s.pdf" % (env_idn))
