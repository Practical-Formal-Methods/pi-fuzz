import re
import logging
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fuzz_config import FUZZ_BUDGET

def poolsize_over_time(env_idn, pool_pop_summ):

    all_size_mean = []
    all_size_std = []
    max_size = 0
    for pp_summ in pool_pop_summ:
        all_sizes = []
        for pp in pp_summ:
            psize = []
            pp = np.array(pp)
            times = pp[:, 1]
            for sec in range(FUZZ_BUDGET):
                cnt = 0
                for tm in times:
                    if tm >= sec:
                        break
                    cnt += 1
                if cnt == 0:
                    psize.append(0)
                else:
                    psize.append(pp[:, 2][cnt-1])

            if max(psize) > max_size:
                max_size = max(psize)

            all_sizes.append(psize)

        all_size_mean.append(np.array(all_sizes).mean(axis=0))
        all_size_std.append(np.array(all_sizes).std(axis=0)) 

    plt.figure(figsize=(10, 7.5))
    ax = plt.subplot(111)
    #ax.spines["top"].set_visible(False)
    #ax.spines["right"].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    #plt.ylim(0, int(max_size + max_size*0.05))

    plt.xticks(range(0, FUZZ_BUDGET+500, 10000), fontsize=16)
    plt.yticks(range(0, int(max_size + max_size*0.05), 100), rotation=30, fontsize=16)

    plt.xlabel("Time (sec)", fontsize=19)
    plt.ylabel("Pool Size", fontsize=19)
    
    labels = ["INC=0.8 INF=0.2", "INC=0.8 INF=0.1", "INC=0.8 INF=0", "INC=0 INF=0.2", "INC=0 INF=0.1", "INC=0 INF=0"]
    colors = ["#3a82b5", "#3f7d48", "#f29544", "#3a82b5", "#3f7d48", "#f29544"]
    linestyles = ["-", "-", "-", "--", "--", "--"]
    for ls, clr, lbl, asm, ass in zip(linestyles, colors, labels, all_size_mean, all_size_std):
        plt.plot(range(FUZZ_BUDGET)[0:FUZZ_BUDGET:60], asm[0:FUZZ_BUDGET:60], ls=ls, lw=2, label=lbl, color=clr)
        plt.fill_between(range(FUZZ_BUDGET)[0:FUZZ_BUDGET:60], asm[0:FUZZ_BUDGET:60]+ass[0:FUZZ_BUDGET:60], asm[0:FUZZ_BUDGET:60]-ass[0:FUZZ_BUDGET:60], color=clr, alpha=0.4)

    plt.legend(loc="lower right", fontsize=16)

    plt.savefig("%s_poolsize_%d.pdf" % (env_idn, FUZZ_BUDGET), bbox_inches="tight")


def sub_warn(pools, mode="num_seeds"):
    all_warn_seed_times = []
    all_num_warns = []
    for pool in pools:
        num_warns = []
        warn_seed_times = []
        for seed in pool:
            if seed.num_warn_mm_hard or seed.num_warn_mm_easy:
                warn_seed_times.append(seed.gen_time)
                num_warns.append(seed.num_warn_mm_hard + seed.num_warn_mm_easy)
        all_warn_seed_times.append(warn_seed_times)
        all_num_warns.append(num_warns)

    all_warns_over_time = []
    for num_warns, ws_times in zip(all_num_warns, all_warn_seed_times):
        warn_over_time = []
        for sec in range(FUZZ_BUDGET):
            if mode == "num_seeds":
                warn_over_time.append(sum(wst < sec for wst in ws_times))
            elif mode == "num_bugs":
                tot_w = 0
                for nw, wst in zip(num_warns, ws_times):
                    if wst < sec:
                        tot_w += nw
                warn_over_time.append(tot_w)
        all_warns_over_time.append(warn_over_time)
    
    return all_warns_over_time


def warn_over_time(env_idn, pools, mode="num_bugs"):  # pools_g, pools_b, pools_gns):

    max_size=0
    all_warns_over_time = []
    all_warns_over_time_mean = []
    all_warns_over_time_std = []
    for pool in pools:
        warns_over_time = sub_warn(pool, mode)
        all_warns_over_time.append(warns_over_time)
        all_warns_over_time_mean.append(np.array(warns_over_time).mean(axis=0))
        all_warns_over_time_std.append(np.array(warns_over_time).std(axis=0))
        
        if len(pool) > max_size:
            max_size = len(pool)

    plt.figure(figsize=(10, 7.5))
    ax = plt.subplot(111)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.xticks(range(0, FUZZ_BUDGET+500, 10000), fontsize=16)
    plt.yticks(fontsize=12, rotation=30)

    plt.xlabel("Time (sec)", fontsize=19)
    if mode == "num_seeds":
        plt.ylabel("#Seeds Found Bug(s)", fontsize=19)
    elif mode == "num_bugs":
        plt.ylabel("# Bugs", fontsize=19)


    labels = ["INC=0.8 INF=0.2", "INC=0.8 INF=0.1", "INC=0.8 INF=0", "INC=0 INF=0.2", "INC=0 INF=0.1", "INC=0 INF=0"]
    colors = ["#3a82b5", "#3f7d48", "#f29544", "#3a82b5", "#3f7d48", "#f29544"]
    linestyles = ["-", "-", "-", "--", "--", "--"]
    for ls, clr, lbl, awm, aws in zip(linestyles, colors, labels, all_warns_over_time_mean, all_warns_over_time_std):
        plt.plot(range(FUZZ_BUDGET)[0:FUZZ_BUDGET:60], awm[0:FUZZ_BUDGET:60], ls=ls, lw=2, label=lbl, color=clr)
        plt.fill_between(range(FUZZ_BUDGET)[0:FUZZ_BUDGET:60], awm[0:FUZZ_BUDGET:60]+aws[0:FUZZ_BUDGET:60], awm[0:FUZZ_BUDGET:60]-aws[0:FUZZ_BUDGET:60], color=clr, alpha=0.4)

    plt.legend(loc="upper left", fontsize=16)
    plt.savefig("%s_warn_overtime_%d_%s.pdf" % (env_idn, FUZZ_BUDGET, mode), bbox_inches="tight")

def boxplot(env_idn, num_tot_warn):
    green_diamond = dict(markerfacecolor="g", marker="D")
    # fig, ax = plt.subplots()
    plt.figure(figsize=(10, 7.5))
    ax = plt.subplot(111)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.xlabel("Fuzzer Settings", fontsize=19)
    plt.ylabel("# Bugs", fontsize=19)
    bplot = ax.boxplot(num_tot_warn, flierprops=green_diamond, patch_artist=True)
    ax.set_xticklabels(["INC=0.8\nINF=0.2", "INC=0.8\nINF=0.1", "INC=0.8\nINF=0", "INC=0\nINF=0.2", "INC=0\nINF=0.1", "INC=0\nINF=0"])

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16, rotation=30)
    colors = ["#3a82b5", "#3f7d48", "#f29544", "#3a82b5", "#3f7d48", "#f29544"]
    linestyles = ["-", "-", "-", "--", "--", "--"]
    for patch, color, ls in zip(bplot['boxes'], colors, linestyles):
        patch.set_facecolor(color)
        patch.set_linestyle(ls)

    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bplot[element], color="black", lw=2)
    plt.setp(bplot["boxes"], lw=2)
    plt.savefig("num_warn_boxplot_%s.pdf" % (env_idn), bbox_inches="tight")


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

    boxplot("all_agents", fuzz_type, all_warns)


def plot_training_curves():
    bipedal_rewards = []
    with open("final_policies/bipedal_monitor.csv", "r") as fr:
        lines = fr.readlines()
        for line in lines[2:]:
            bipedal_rewards.append(float(line.split(",")[0]))

    lunar_rewards = []
    with open("final_policies/lunar_monitor.csv", "r") as fr:
        lines = fr.readlines()
        for line in lines[2:]:
            lunar_rewards.append(float(line.split(",")[0]))

    linetrack_rewards = []
    with open("final_policies/linetrack_monitor.csv", "r") as fr:
        lines = fr.readlines()
        for line in lines:
            if line == "\n": continue
            linetrack_rewards.append(float(line.split("\t")[1].split(" ")[2]))

    data = [bipedal_rewards, lunar_rewards, linetrack_rewards]
    labels = ["BipedalWalker", "LunarLander", "Linetrack"]
    colors = ["#3a82b5", "#3a82b5", "#3a82b5"]
    # colors = ["#393bbd", "#c41b1b", "#32e6e6"]

    plt.figure(figsize=(12, 15))
    fig, (ax0, ax1, ax2) = plt.subplots(3, figsize=(11,11))
    axes = [ax0, ax1, ax2]
    for ax, clr, lbl, dt in zip(axes, colors, labels, data):
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_title(lbl, fontsize=15)
        ax.plot(dt, ls="-", lw=2, label=lbl, color=clr)

    plt.xlabel("Iteration", fontsize=18)
    ax1.set_ylabel("Rewards", fontsize=18)
    plt.savefig("training_curves.pdf", bbox_inches="tight")

    # plt.legend(loc="upper left", fontsize=12)
