import pickle
import numpy as np
import EnvWrapper as EW
import matplotlib.pyplot as plt
from fuzz_config import FUZZ_BUDGET

lunar_files = [
    "extras/lunar_results/lunar_gbox_1_20210905_130043_sp0.200000.p",
    "extras/lunar_results/lunar_gbox_2_20210905_130047_sp0.200000.p",
    "extras/lunar_results/lunar_gbox_3_20210905_130051_sp0.200000.p",
    "extras/lunar_results/lunar_gbox_4_20210905_130057_sp0.200000.p",
    "extras/lunar_results/lunar_gbox_5_20210905_130101_sp0.200000.p",
    "extras/lunar_results/lunar_gbox_6_20210905_130104_sp0.200000.p",
    "extras/lunar_results/lunar_gbox_7_20210905_130113_sp0.200000.p",
    "extras/lunar_results/lunar_gbox_8_20210905_130116_sp0.200000.p",
]

bipedal_files = [
    "extras/bipedal_results/bipedal_gbox_1_20210822_133532_sp0.200000.p",
    "extras/bipedal_results/bipedal_gbox_2_20210822_133544_sp0.200000.p",
    "extras/bipedal_results/bipedal_gbox_3_20210822_133558_sp0.200000.p",
    "extras/bipedal_results/bipedal_gbox_4_20210822_133607_sp0.200000.p",
    "extras/bipedal_results/bipedal_gbox_5_20210822_133618_sp0.200000.p",
    "extras/bipedal_results/bipedal_gbox_6_20210822_133631_sp0.200000.p",
    "extras/bipedal_results/bipedal_gbox_7_20210822_133638_sp0.200000.p",
    "extras/bipedal_results/bipedal_gbox_8_20210822_133651_sp0.200000.p",
]

linetrack_files = [
       "extras/linetrack_results/linetrack_gbox_1_20210904_001206_sp0.200000.p", 
       "extras/linetrack_results/linetrack_gbox_2_20210904_001222_sp0.200000.p", 
       "extras/linetrack_results/linetrack_gbox_3_20210904_001309_sp0.200000.p", 
       "extras/linetrack_results/linetrack_gbox_4_20210904_001318_sp0.200000.p", 
       "extras/linetrack_results/linetrack_gbox_5_20210904_001338_sp0.200000.p", 
       "extras/linetrack_results/linetrack_gbox_6_20210904_001353_sp0.200000.p", 
       "extras/linetrack_results/linetrack_gbox_7_20210904_001406_sp0.200000.p", 
       "extras/linetrack_results/linetrack_gbox_8_20210904_001417_sp0.200000.p", 
]

env_iden = "linetrack"
agent_path = "policies/linetrack_org.pth"
all_baseline_times = []
all_pifuzz_times = []
for idx, lfile in enumerate(linetrack_files):
    r_seed = idx
    game = EW.Wrapper(env_iden)
    game.create_environment(env_seed=r_seed)
    game.create_model(agent_path, r_seed)
    
    pop_summ, pool, tot_warns_mm_e, tot_warns_mm_h, num_cycles, total_trials, total_time, avg_time = pickle.load(open(lfile, "rb"))
    bline_warn_times = []
    pifuzz_warn_times = []

    rews = []
    for seed in pool:
        if env_iden == "linetrack":
            exp_rng = np.random.default_rng(r_seed)
            game.env.reset(exp_rng)
        else:
            game.env.seed(r_seed)

        game.set_state(seed.hi_lvl_state) 
        agent_reward, _, _ = game.run_pol_fuzz(seed.data, "qualitative")

        if agent_reward == 0:
            bline_warn_times.append(seed.gen_time)
        if agent_reward == 0 and seed.num_warn_mm_hard > 0:
            pifuzz_warn_times.append(seed.gen_time)
    
        rews.append(agent_reward)

    print(len(pool), len(bline_warn_times))
    print(len(pool), len(pifuzz_warn_times))

    all_baseline_times.append(bline_warn_times)
    all_pifuzz_times.append(pifuzz_warn_times)

all_bline_warn_over_times = []
all_pifuzz_warn_over_times = []
for pi_w_times, bline_w_times in zip(all_pifuzz_times, all_baseline_times):
    pifuzz_warns_over_time = []
    bline_warns_over_time = []
    for sec in range(FUZZ_BUDGET):
        pifuzz_warns_over_time.append(sum(wst < sec for wst in pi_w_times))
        bline_warns_over_time.append(sum(wst < sec for wst in bline_w_times))

    all_pifuzz_warn_over_times.append(pifuzz_warns_over_time)
    all_bline_warn_over_times.append(bline_warns_over_time)

awm_bline = np.array(all_bline_warn_over_times).mean(axis=0)
aws_bline = np.array(all_bline_warn_over_times).std(axis=0)
awm_pifuzz = np.array(all_pifuzz_warn_over_times).mean(axis=0)
aws_pifuzz = np.array(all_pifuzz_warn_over_times).std(axis=0)

plt.figure(figsize=(10, 7.5))
ax = plt.subplot(111)

ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

plt.xticks(range(0, FUZZ_BUDGET+500, 10000), fontsize=16)
plt.yticks(fontsize=12, rotation=30)

plt.xlabel("Time (sec)", fontsize=19)
plt.ylabel("#Seeds Found Bug(s)", fontsize=19)

labels = ["Pifuzz", "Baseline"]
colors = ["#3a82b5", "#3f7d48"]
linestyles = ["-", "--"]

plt.plot(range(FUZZ_BUDGET)[0:FUZZ_BUDGET:60], awm_pifuzz[0:FUZZ_BUDGET:60], ls='-', lw=2, label='Pifuzz', color='#3a82b5')
plt.fill_between(range(FUZZ_BUDGET)[0:FUZZ_BUDGET:60], awm_pifuzz[0:FUZZ_BUDGET:60]+aws_pifuzz[0:FUZZ_BUDGET:60], awm_pifuzz[0:FUZZ_BUDGET:60]-aws_pifuzz[0:FUZZ_BUDGET:60], color='#3a82b5', alpha=0.4)

plt.plot(range(FUZZ_BUDGET)[0:FUZZ_BUDGET:60], awm_bline[0:FUZZ_BUDGET:60], ls='--', lw=2, label='Baseline', color='#3f7d48')
plt.fill_between(range(FUZZ_BUDGET)[0:FUZZ_BUDGET:60], awm_bline[0:FUZZ_BUDGET:60]+aws_bline[0:FUZZ_BUDGET:60], awm_bline[0:FUZZ_BUDGET:60]-aws_bline[0:FUZZ_BUDGET:60], color='#3f7d48', alpha=0.4)

plt.legend(loc="upper left", fontsize=16)
plt.savefig("%s_warn_overtime_%d_%s.pdf" % ("linetrack", FUZZ_BUDGET, "bline"), bbox_inches="tight")
