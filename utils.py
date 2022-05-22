import re
import os
import logging
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import FUZZ_BUDGET

def process_pools(pools):
    largest_pool_size = 0
    for pool in pools:
        if len(pool) > largest_pool_size: largest_pool_size = len(pool)

    all_bugs = []
    for cur_pool in pools:
        cum_bugs_lst, cum_bugs = [], 0
        for i in range(largest_pool_size):
            # fill the rest of the cum_bugs_lst with latest cum_bugs value
            if i < len(cur_pool): 
                fuzz_seed = cur_pool[i]
                cum_bugs += fuzz_seed.num_bugs
            cum_bugs_lst.append(cum_bugs)

    all_bugs.append(cum_bugs_lst)
    mean_bugs = np.array(all_bugs).mean(axis=0)  # mean bugs over pool size
    std_bugs = np.array(all_bugs).std(axis=0)  # std bugs over pool size

    return mean_bugs, std_bugs


def orcl_evl_bipedal():
    mmseedbugbasic_pools, mmseedbugext_pools, failseedbug_pools, ruleseedbug_pools = [], [], [], []

    mmsbb_file_templ = 'Ebipedal_R\d+_Ommseedbugbasic_Finc_C2.0_I0.1_\d+_\d+.p'
    mmsbe_file_templ = 'Ebipedal_R\d+_Ommseedbugext_Finc_C2.0_I0.1_\d+_\d+.p'
    fsb_file_templ = 'Ebipedal_R\d+_Ofailseedbug_Finc_C2.0_I0.1_\d+_\d+.p'
    rsb_file_templ = 'Ebipedal_R\d+_Oruleseedbug_Finc_C2.0_I0.1_\d+_\d+.p'
    
    mmsbb_files, mmsbe_files, fsb_files, rsb_files = [], [], [], []
    for fname in os.listdir("pifuzz_logs"):
        if re.search(mmsbb_file_templ, fname):
            mmsbb_files.append("pifuzz_logs/" + fname)
        if re.search(mmsbe_file_templ, fname):
            mmsbe_files.append("pifuzz_logs/" + fname)
        if re.search(fsb_file_templ, fname):
            fsb_files.append("pifuzz_logs/" + fname)
        if re.search(rsb_file_templ, fname):
            rsb_files.append("pifuzz_logs/" + fname)

    for mmsbb, mmsbe, fsb, rsb in zip(mmsbb_files, mmsbe_files, fsb_files, rsb_files):
        mmseedbugbasic_pools.append(mmsbb[1])
        mmseedbugext_pools.append(mmsbe[1])
        failseedbug_pools.append(fsb[1])
        ruleseedbug_pools.append(rsb[1])
    
    mmsbb_means, mmsbb_stds = process_pools(mmseedbugbasic_pools)
    mmsbe_means, mmsbe_stds = process_pools(mmseedbugext_pools)
    fsb_means, fsb_stds = process_pools(failseedbug_pools)
    rsb_means, rsb_stds = process_pools(ruleseedbug_pools)

    labels = ["MMSeedBugExt", "FailureSeedBug", "MMSeedBugBasic", "RuleSeedBug"]
    colors = ["#344588", "#f29544", "#e32d2d", "#955a92" ]

    plt.figure(figsize=(10, 7.5))
    ax = plt.subplot(111)
    
    ax.set_yscale('log')
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    pool_size = len(mmsbb_means[0])
    plt.xticks(range(0, pool_size+10, 200), fontsize=16)
    plt.yticks(fontsize=12, rotation=30)
    plt.xlabel("Pool Size", fontsize=19)
    plt.ylabel("# Bugs", fontsize=19)

    for mean, std, lbl, clr in zip ([mmsbe_means, fsb_means, mmsbb_means, rsb_means], [mmsbe_stds, fsb_stds, mmsbb_stds, rsb_stds], labels, colors):
        plt.plot(range(pool_size), mean, lw=2, label=lbl, color=clr)
        plt.fill_between(range(pool_size), mean+std, mean-std, color=clr, alpha=0.3)
    
    plt.legend(loc="lower right", fontsize=16 )

    plt.savefig("bipedal_bugs_pool_size.pdf", bbox_inches="tight")


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


def sub_warn(pools, mode):
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
    for clr, lbl, awm, aws in zip(colors, labels, all_warns_over_time_mean, all_warns_over_time_std):
        plt.plot(range(FUZZ_BUDGET)[0:FUZZ_BUDGET:60], awm[0:FUZZ_BUDGET:60], lw=2, label=lbl, color=clr)
        plt.fill_between(range(FUZZ_BUDGET)[0:FUZZ_BUDGET:60], awm[0:FUZZ_BUDGET:60]+aws[0:FUZZ_BUDGET:60], awm[0:FUZZ_BUDGET:60]-aws[0:FUZZ_BUDGET:60], color=clr, alpha=0.4)

    plt.legend(loc="upper left", fontsize=16)
    plt.savefig("%s_warn_overtime_%d_%s.pdf" % (env_idn, FUZZ_BUDGET, mode), bbox_inches="tight")



def plot_orcl_evl(pools, fail_pools, perf_pools, env_idn):  # pools  orresponds to 8 run of one setting INC, INF02
    
    max_pool_size = 0
    for pool in pools:
        if len(pool) > max_pool_size: max_pool_size = len(pool)

    print(max_pool_size)
    print(len(pools))
    print(len(fail_pools))
    print(len(perf_pools))
    
    all_mm_base_warns = []
    all_mm_ext_warns = []
    all_fail_warns = []
    all_rule_warns = []
    all_opt_warns = []
    all_idl_warns = []
    all_f_idl_warns = []
    all_p_idl_warns = []
    for pool, fail_pool, perf_pool in zip(pools, fail_pools, perf_pools):
        mm_base_warns = []
        mm_ext_warns = []
        fail_warns = []
        rule_warns = []
        opt_warns = []
        idl_warns = []
        f_idl_warns = []
        p_idl_warns = []
        seed_ptr = 0
        num_mm_base_warn = 0
        num_mm_ext_warn = 0
        num_fail_warn = 0
        num_rule_warn = 0
        num_opt_warn = 0
        num_idl_warn = 0
        num_f_idl_warn = 0
        num_p_idl_warn = 0
        '''
        for time_bucket in range(0, FUZZ_BUDGET, 60):
            seed = pool[seed_ptr]
            while seed.gen_time < time_bucket:
                num_mm_base_warn += seed.num_warn_mm_hard
                num_mm_ext_warn += seed.num_warn_mm_hard + seed.num_warn_mm_easy
                num_fail_warn += seed.num_warn_fail
                num_rule_warn += seed.num_warn_rule
                num_opt_warn += seed.num_warn_optimal
                if not env_idn == "bipedal": num_idl_warn += seed.num_warn_ideal

                seed_ptr += 1
                seed = pool[seed_ptr]
        '''
        for fseed, fseed_fail, fseed_perf in zip(pool, fail_pool, perf_pool):
            num_mm_base_warn += fseed.num_warn_mm_hard
            num_mm_ext_warn += fseed.num_warn_mm_hard + fseed.num_warn_mm_easy
            num_fail_warn += fseed.num_warn_fail
            num_rule_warn += fseed.num_warn_rule
            num_opt_warn += fseed.num_warn_optimal
            if not env_idn == "bipedal": 
                num_idl_warn += fseed.num_warn_ideal
                num_f_idl_warn += fseed_fail.num_warn_ideal
            
            if env_idn == "linetrack":
                num_p_idl_warn += fseed_perf.num_warn_ideal
            
            mm_base_warns.append(num_mm_base_warn)
            mm_ext_warns.append(num_mm_ext_warn)
            fail_warns.append(num_fail_warn)
            rule_warns.append(num_rule_warn)
            opt_warns.append(num_opt_warn)
            idl_warns.append(num_idl_warn)
            f_idl_warns.append(num_f_idl_warn)
            p_idl_warns.append(num_p_idl_warn)
        
        for _ in range(max_pool_size - len(pool)):
            mm_base_warns.append(num_mm_base_warn)
            mm_ext_warns.append(num_mm_ext_warn)
            fail_warns.append(num_fail_warn)
            rule_warns.append(num_rule_warn)
            opt_warns.append(num_opt_warn)
            idl_warns.append(num_idl_warn)
            f_idl_warns.append(num_f_idl_warn)
            p_idl_warns.append(num_p_idl_warn)
            
        all_mm_base_warns.append(mm_base_warns)
        all_mm_ext_warns.append(mm_ext_warns)
        all_fail_warns.append(fail_warns)
        all_rule_warns.append(rule_warns)
        all_opt_warns.append(opt_warns)
        all_idl_warns.append(idl_warns)
        all_f_idl_warns.append(f_idl_warns)
        all_p_idl_warns.append(p_idl_warns)
    
    print(np.array(all_f_idl_warns).shape)
    all_warns_mean = [ np.array(all_mm_ext_warns).mean(axis=0), np.array(all_p_idl_warns).mean(axis=0), np.array(all_fail_warns).mean(axis=0), np.array(all_f_idl_warns).mean(axis=0), np.array(all_opt_warns).mean(axis=0), np.array(all_mm_base_warns).mean(axis=0), np.array(all_idl_warns).mean(axis=0), np.array(all_rule_warns).mean(axis=0) ]
    all_warns_std = [ np.array(all_mm_ext_warns).std(axis=0), np.array(all_p_idl_warns).std(axis=0), np.array(all_fail_warns).std(axis=0), np.array(all_f_idl_warns).std(axis=0), np.array(all_opt_warns).std(axis=0), np.array(all_mm_base_warns).std(axis=0), np.array(all_idl_warns).std(axis=0), np.array(all_rule_warns).std(axis=0) ]

    labels = [ "MMSeedBugExt", "PerfectBug", "FailureSeedBug", "MMBug", "PerfectSeedBug", "MMSeedBugBasic", "MMSeedBug2Bug", "RuleSeedBug" ]
    # labels = ["ExtendedMM", "Failure-Based", "FailProb.", "Perfect", "BasicMM", "BasicMMFailProb.", "Rule-Based" ]
    colors = ["#344588", "#f963e5", "#f29544", "#b2b200", "#33fff6", "#e32d2d", "#43ce3b", "#955a92" ]
    #0213c7  e32dd2  
    
    bug_labels = ["PerfectBug", "MMBug", "MMSeedBug2Bug"]
    seedbug_labels = ["MMSeedBugExt", "FailureSeedBug", "PerfectSeedBug", "MMSeedBugBasic", "RuleSeedBug" ]
    
    plt.figure(figsize=(10, 7.5))
    ax = plt.subplot(111)
    
    ax.set_yscale('log')
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.xticks(range(0, max_pool_size+10, 200), fontsize=16)
    plt.yticks(fontsize=12, rotation=30)
    plt.xlabel("Pool Size", fontsize=19)
    plt.ylabel("# Bugs", fontsize=19)

    if env_idn == "lunar":
        colors[5], colors[6] = colors[6], colors[5]
        labels[5], labels[6] = labels[6], labels[5]
        all_warns_mean[5], all_warns_mean[6] = all_warns_mean[6], all_warns_mean[5]
        all_warns_std[5], all_warns_std[6] = all_warns_std[6], all_warns_std[5]

    if env_idn == "linetrack":
        colors[2], colors[3] = colors[3], colors[2]
        labels[2], labels[3] = labels[3], labels[2]
        all_warns_mean[2], all_warns_mean[3] = all_warns_mean[3], all_warns_mean[2]
        all_warns_std[2], all_warns_std[3] = all_warns_std[3], all_warns_std[2]

    for clr, lbl, awm, aws in zip(colors, labels, all_warns_mean, all_warns_std):
        alph = 0.4
        #if lbl in seedbug_labels : continue
        if lbl in bug_labels : continue
        if not env_idn == "linetrack" and lbl == "PerfectSeedBug": continue  # optimal oracle available only in linetrack
        if not env_idn == "linetrack" and lbl == "PerfectBug": continue  # optimal oracle available only in linetrack
        if (env_idn == "bipedal" and lbl == "MMSeedBug2Bug") or (env_idn == "bipedal" and lbl == "MMBug"): continue
        if env_idn == "lunar" and lbl == "RuleSeedBug": aws = [ min(s, m/1.5) for m, s in zip(awm, aws)]
        if env_idn == "lunar" and lbl == "MMSeedBugExt": aws = [ min(s, m/1.5) for m, s in zip(awm, aws)] 
        # aws = min(aws, awm/2)
        # range(0, FUZZ_BUDGET, 60)
        plt.plot(range(max_pool_size), awm, lw=2, label=lbl, color=clr)
        plt.fill_between(range(max_pool_size), awm+aws, awm-aws, color=clr, alpha=alph)

    if env_idn == "lunar":
        handles, labels = plt.gca().get_legend_handles_labels()
        # order = [2, 3, 0, 1, 4, 5]
        order1 = [2, 3]
        order2 = [0, 1, 4, 5]
        # plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc="lower right", fontsize=16 )
        
        first_legend = ax.legend( [handles[idx] for idx in order1], [labels[idx] for idx in order1], loc='lower left', bbox_to_anchor=(0.25, 0.), fontsize=16)
        ax.add_artist(first_legend)
        ax.legend( [handles[idx] for idx in order2], [labels[idx] for idx in order2], loc='lower right', fontsize=16)

    elif env_idn == "linetrack":
        handles, labels = plt.gca().get_legend_handles_labels()
        # order = [1, 2, 6, 0, 3, 4, 5, 7]
        # order1 = [1, 2, 6]
        # order2 = [0, 3, 4, 5, 7]
        bug_order = [0, 1, 2] 
        seedbug_order = [0, 1, 2, 3, 4]
        print(labels)
        #first_legend = ax.legend( [handles[idx] for idx in order1], [labels[idx] for idx in order1], loc='lower left', bbox_to_anchor=(0.25, 0.), fontsize=16)
        #ax.add_artist(first_legend)
        #ax.legend( [handles[idx] for idx in order2], [labels[idx] for idx in order2], loc='lower right', fontsize=16)
         
        #ax.legend( [handles[idx] for idx in bug_order], [labels[idx] for idx in bug_order], loc='lower right', fontsize=16)
        ax.legend( [handles[idx] for idx in seedbug_order], [labels[idx] for idx in seedbug_order], loc='lower right', fontsize=16)
        
    else:
        plt.legend(loc="lower right", fontsize=16 )
        # plt.legend(loc="upper left", fontsize=16 )

    plt.savefig("%s_oracle_eval_seedbug_psize.pdf" % (env_idn), bbox_inches="tight")



def plot_fuzzer_evl(env_idn, cov0_inf0_pools, cov0_inf02_pools, inf0_pools, inf02_pools):

    all_pools = [cov0_inf0_pools, cov0_inf02_pools, inf0_pools, inf02_pools]

    all_setting_max_psize = []
    all_setting_mean_warns = []
    all_setting_std_warns = []
    for pools in all_pools:
        all_mm_base_warns = []
        all_mm_ext_warns = []

        max_pool_size = 0
        for pool in pools:
            if len(pool) > max_pool_size: max_pool_size = len(pool)
        
        print(max_pool_size)
        for pool in pools:
            mm_base_warns = []
            mm_ext_warns = []
            num_mm_base_warn = 0
            num_mm_ext_warn = 0
            
            for fseed in pool:
                num_mm_base_warn += fseed.num_warn_mm_hard
                num_mm_ext_warn += fseed.num_warn_mm_hard + fseed.num_warn_mm_easy

                mm_base_warns.append(num_mm_base_warn)
                mm_ext_warns.append(num_mm_ext_warn)
            
            for _ in range(max_pool_size - len(pool)):
                mm_base_warns.append(num_mm_base_warn)
                mm_ext_warns.append(num_mm_ext_warn)

            all_mm_base_warns.append(mm_base_warns)
            all_mm_ext_warns.append(mm_ext_warns)
        
        all_setting_max_psize.append(max_pool_size)
        all_setting_max_psize.append(max_pool_size)  # extra is neede for base mm

        all_setting_mean_warns.append( np.array(all_mm_ext_warns).mean(axis=0))
        all_setting_std_warns.append( np.array(all_mm_ext_warns).std(axis=0))
        all_setting_mean_warns.append( np.array(all_mm_base_warns).mean(axis=0))
        all_setting_std_warns.append( np.array(all_mm_base_warns).std(axis=0))
    
    labels = ["No Thold, InfP=0, E.MM", "No Thold, InfP=0, B.MM", "No Thold, InfP=0.2, E.MM", "No Thold, InfP=0.2, B.MM", "InfP=0, E.MM", "InfP=0, B.MM", "InfP=0.2, E.MM", "InfP=0.2, B.MM" ]
    colors = ["#0213c7", "#0213c7", "#3f7d48", "#3f7d48", "#f29544", "#f29544", "#33fff6", "#33fff6" ]
    linestyles = ["-", "--", "-", "--", "-", "--", "-", "--"]

    plt.figure(figsize=(10, 7.5))
    ax = plt.subplot(111)
    
    ax.set_yscale('log')
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.xticks(range(0, max(all_setting_max_psize), 200), fontsize=16)
    plt.yticks(fontsize=12, rotation=30)
    plt.xlabel("Pool Size", fontsize=19)
    plt.ylabel("# Bugs", fontsize=19)

    for max_psize, ls, clr, lbl, awm, aws in zip(all_setting_max_psize, linestyles, colors, labels, all_setting_mean_warns, all_setting_std_warns):
        plt.plot(range(max_psize), awm, ls=ls, lw=2, label=lbl, color=clr)
        plt.fill_between(range(max_psize), awm+aws, awm-aws, color=clr, alpha=0.4)

    plt.legend(loc="lower right", fontsize=16)
    plt.savefig("%s_fuzzer_eval_psize.pdf" % (env_idn), bbox_inches="tight")


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

    eplt.xticks(fontsize=16)
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
