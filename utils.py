import re
import os
import pickle
import logging
import matplotlib
# these parameters are needs to be set, for camera ready plots
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import numpy as np
import matplotlib.pyplot as plt

def plot(benchmark, mean_data, std_data, labels, colors):

    plt.figure(figsize=(10, 7.5))
    ax = plt.subplot(111)
    
    ax.set_yscale('log')
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    pool_size = len(mean_data[0])
    plt.xticks(range(0, pool_size+10, 200), fontsize=16)
    plt.yticks(fontsize=12, rotation=30)
    plt.xlabel("Pool Size", fontsize=19)
    plt.ylabel("# Bugs", fontsize=19)
    
    for mean, std, lbl, clr in zip (mean_data, std_data, labels, colors):
        plt.plot(range(pool_size), mean, lw=2, label=lbl, color=clr)
        plt.fill_between(range(pool_size), mean+std, mean-std, color=clr, alpha=0.3)
    
    if benchmark == "bipedal":
        plt.legend(loc="lower right", fontsize=16, shadow=True)
        plt.savefig("%s_bugs_pool_size.pdf" % benchmark, bbox_inches="tight")
        return 
    
    handles, labels = plt.gca().get_legend_handles_labels()
    if benchmark == "lunar":
        order1 = [0, 1]
        order2 = [2, 3, 4, 5]
    elif benchmark == "highway":
        order1 = [0, 1, 2]
        order2 = [3, 4, 5, 6, 7]
   
    first_legend = ax.legend( [handles[idx] for idx in order1], [labels[idx] for idx in order1], loc='lower left', bbox_to_anchor=(0.25, 0.), fontsize=16, shadow=True)
    ax.add_artist(first_legend)
    ax.legend( [handles[idx] for idx in order2], [labels[idx] for idx in order2], loc='lower right', fontsize=16, shadow=True)
        
    plt.savefig("%s_bugs_pool_size.pdf" % benchmark, bbox_inches="tight")


def process_pools(pools):
    largest_pool_size = 0
    for pool in pools:
        if len(pool) > largest_pool_size: largest_pool_size = len(pool)

    # largest_pool_size = 1262
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

    mmsbb_file_templ = 'Ebipedal_R\d+_Ommseedbugbasic_Finc_C2.0_I0.2_\d+_\d+.p'
    mmsbe_file_templ = 'Ebipedal_R\d+_Ommseedbugext_Finc_C2.0_I0.2_\d+_\d+.p'
    fsb_file_templ = 'Ebipedal_R\d+_Ofailseedbug_Finc_C2.0_I0.2_\d+_\d+.p'
    rsb_file_templ = 'Ebipedal_R\d+_Oruleseedbug_Finc_C2.0_I0.2_\d+_\d+.p'
    
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

        mmseedbugbasic_pools.append(pickle.load(open(mmsbb, "rb"))[1])
        mmseedbugext_pools.append(pickle.load(open(mmsbe, "rb"))[1])
        failseedbug_pools.append(pickle.load(open(fsb, "rb"))[1])
        ruleseedbug_pools.append(pickle.load(open(rsb, "rb"))[1])
    
    mmsbb_means, mmsbb_stds = process_pools(mmseedbugbasic_pools)
    mmsbe_means, mmsbe_stds = process_pools(mmseedbugext_pools)
    fsb_means, fsb_stds = process_pools(failseedbug_pools)
    rsb_means, rsb_stds = process_pools(ruleseedbug_pools)
    
    mean_data = [mmsbe_means, fsb_means, mmsbb_means, rsb_means]
    std_data = [mmsbe_stds, fsb_stds, mmsbb_stds, rsb_stds]
    labels = ["MMSeedBugExt", "FailureSeedBug", "MMSeedBugBasic", "RuleSeedBug"]
    colors = ["#344588", "#f29544", "#e32d2d", "#955a92" ]

    plot("bipedal", mean_data, std_data, labels, colors)

def orcl_evl_lunar():
    mmseedbugbasic_pools, mmseedbug2bug_pools, mmseedbugext_pools, mmbug_pools,failseedbug_pools, ruleseedbug_pools = [], [], [], [], [], []

    mmsbb_file_templ = 'Elunar_R\d+_Ommseedbugbasic_Finc_C0.65_I0.2_\d+_\d+.p'
    mmsbe_file_templ = 'Elunar_R\d+_Ommseedbugext_Finc_C0.65_I0.2_\d+_\d+.p'
    mmb_file_templ = 'Elunar_R\d+_Ommbug_Finc_C0.65_I0.2_\d+_\d+.p'
    mmsb2b_file_templ = 'Elunar_R\d+_Ommseedbug2bug_Finc_C0.65_I0.2_\d+_\d+.p'
    fsb_file_templ = 'Elunar_R\d+_Ofailseedbug_Finc_C0.65_I0.2_\d+_\d+.p'
    rsb_file_templ = 'Elunar_R\d+_Oruleseedbug_Finc_C0.65_I0.2_\d+_\d+.p'
    
    mmsbb_files, mmsb2b_files, mmsbe_files, mmb_files, fsb_files, rsb_files = [], [], [], [], [], []
    for fname in os.listdir("pifuzz_logs"):
        if re.search(mmsbb_file_templ, fname):
            mmsbb_files.append("pifuzz_logs/" + fname)
        if re.search(mmsb2b_file_templ, fname):
            mmsb2b_files.append("pifuzz_logs/" + fname)
        if re.search(mmsbe_file_templ, fname):
            mmsbe_files.append("pifuzz_logs/" + fname)
        if re.search(mmb_file_templ, fname):
            mmb_files.append("pifuzz_logs/" + fname)
        if re.search(fsb_file_templ, fname):
            fsb_files.append("pifuzz_logs/" + fname)
        if re.search(rsb_file_templ, fname):
            rsb_files.append("pifuzz_logs/" + fname)

    for mmsbb, mmsb2b, mmsbe, mmb, fsb, rsb in zip(mmsbb_files, mmsb2b_files, mmsbe_files, mmb_files, fsb_files, rsb_files):

        mmseedbugbasic_pools.append(pickle.load(open(mmsbb, "rb"))[1])
        mmseedbug2bug_pools.append(pickle.load(open(mmsb2b, "rb"))[1])
        mmseedbugext_pools.append(pickle.load(open(mmsbe, "rb"))[1])
        mmbug_pools.append(pickle.load(open(mmb, "rb"))[1])
        failseedbug_pools.append(pickle.load(open(fsb, "rb"))[1])
        ruleseedbug_pools.append(pickle.load(open(rsb, "rb"))[1])
    
    mmsbb_means, mmsbb_stds = process_pools(mmseedbugbasic_pools)
    mmsb2b_means, mmsb2b_stds = process_pools(mmseedbug2bug_pools)
    mmsbe_means, mmsbe_stds = process_pools(mmseedbugext_pools)
    mmb_means, mmb_stds = process_pools(mmbug_pools)
    fsb_means, fsb_stds = process_pools(failseedbug_pools)
    rsb_means, rsb_stds = process_pools(ruleseedbug_pools)
    
    mean_data = [mmb_means, mmsb2b_means, mmsbe_means, fsb_means, mmsbb_means, rsb_means]
    std_data = [mmb_stds, mmsb2b_stds, mmsbe_stds, fsb_stds, mmsbb_stds, rsb_stds]
    labels = ["MMBug", "MMSeedBug2Bug", "MMSeedBugExt", "FailureSeedBug", "MMSeedBugBasic", "RuleSeedBug"]
    colors = ["#b2b200", "#43ce3b", "#344588", "#f29544", "#e32d2d", "#955a92" ]

    plot("lunar", mean_data, std_data, labels, colors)


def orcl_evl_highway():
    mmseedbugbasic_pools, mmseedbug2bug_pools, mmseedbugext_pools, mmbug_pools,failseedbug_pools, ruleseedbug_pools, perfbug_pools, perfseedbug_pools = [], [], [], [], [], [], [], []

    mmsbb_file_templ = 'Ehighway_R\d+_Ommseedbugbasic_Finc_C3.6_I0.2_\d+_\d+.p'
    mmsbe_file_templ = 'Ehighway_R\d+_Ommseedbugext_Finc_C3.6_I0.2_\d+_\d+.p'
    mmb_file_templ = 'Ehighway_R\d+_Ommbug_Finc_C3.6_I0.2_\d+_\d+.p'
    mmsb2b_file_templ = 'Ehighway_R\d+_Ommseedbug2bug_Finc_C3.6_I0.2_\d+_\d+.p'
    fsb_file_templ = 'Ehighway_R\d+_Ofailseedbug_Finc_C3.6_I0.2_\d+_\d+.p'
    rsb_file_templ = 'Ehighway_R\d+_Oruleseedbug_Finc_C3.6_I0.2_\d+_\d+.p'
    pb_file_templ = 'Ehighway_R\d+_Operfbug_Finc_C3.6_I0.2_\d+_\d+.p'
    psb_file_templ = 'Ehighway_R\d+_Operfseedbug_Finc_C3.6_I0.2_\d+_\d+.p'
    
    mmsbb_files, mmsb2b_files, mmsbe_files, mmb_files, fsb_files, rsb_files, pb_files, psb_files = [], [], [], [], [], [], [], []
    for fname in os.listdir("pifuzz_logs"):
        if re.search(mmsbb_file_templ, fname):
            mmsbb_files.append("pifuzz_logs/" + fname)
        if re.search(mmsb2b_file_templ, fname):
            mmsb2b_files.append("pifuzz_logs/" + fname)
        if re.search(mmsbe_file_templ, fname):
            mmsbe_files.append("pifuzz_logs/" + fname)
        if re.search(mmb_file_templ, fname):
            mmb_files.append("pifuzz_logs/" + fname)
        if re.search(fsb_file_templ, fname):
            fsb_files.append("pifuzz_logs/" + fname)
        if re.search(rsb_file_templ, fname):
            rsb_files.append("pifuzz_logs/" + fname)
        if re.search(pb_file_templ, fname):
            pb_files.append("pifuzz_logs/" + fname)
        if re.search(psb_file_templ, fname):
            psb_files.append("pifuzz_logs/" + fname)

    for mmsbb, mmsb2b, mmsbe, mmb, fsb, rsb, pb, psb in zip(mmsbb_files, mmsb2b_files, mmsbe_files, mmb_files, fsb_files, rsb_files, pb_files, psb_files):

        mmseedbugbasic_pools.append(pickle.load(open(mmsbb, "rb"))[1])
        mmseedbug2bug_pools.append(pickle.load(open(mmsb2b, "rb"))[1])
        mmseedbugext_pools.append(pickle.load(open(mmsbe, "rb"))[1])
        mmbug_pools.append(pickle.load(open(mmb, "rb"))[1])
        failseedbug_pools.append(pickle.load(open(fsb, "rb"))[1])
        ruleseedbug_pools.append(pickle.load(open(rsb, "rb"))[1])
        perfbug_pools.append(pickle.load(open(pb, "rb"))[1])
        perfseedbug_pools.append(pickle.load(open(psb, "rb"))[1])
    
    mmsbb_means, mmsbb_stds = process_pools(mmseedbugbasic_pools)
    mmsb2b_means, mmsb2b_stds = process_pools(mmseedbug2bug_pools)
    mmsbe_means, mmsbe_stds = process_pools(mmseedbugext_pools)
    mmb_means, mmb_stds = process_pools(mmbug_pools)
    fsb_means, fsb_stds = process_pools(failseedbug_pools)
    rsb_means, rsb_stds = process_pools(ruleseedbug_pools)
    pb_means, pb_stds = process_pools(perfbug_pools)
    psb_means, psb_stds = process_pools(perfseedbug_pools)
    
    mean_data = [pb_means, mmb_means, mmsb2b_means, mmsbe_means, fsb_means, psb_means, mmsbb_means, rsb_means]
    std_data = [pb_stds, mmb_stds, mmsb2b_stds, mmsbe_stds, fsb_stds, psb_stds, mmsbb_stds, rsb_stds]
    labels = ["PerfectBug", "MMBug", "MMSeedBug2Bug", "MMSeedBugExt", "FailureSeedBug", "PerfectSeedBug", "MMSeedBugBasic", "RuleSeedBug"]
    colors = ["#f963e5", "#b2b200", "#43ce3b", "#344588", "#f29544", "#33fff6", "#e32d2d", "#955a92" ]

    plot("highway", mean_data, std_data, labels, colors)


def setup_logger(name, log_file, level=logging.DEBUG):
    handler = logging.FileHandler(log_file, mode="w")
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger