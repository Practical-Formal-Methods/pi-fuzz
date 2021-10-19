import pickle
import numpy as np
import EnvWrapper as EW
import matplotlib.pyplot as plt
from fuzz_config import FUZZ_BUDGET

agent_path = "policies/lunar_org"
lunar_files = [
    "lunar_gbox_1_20210905_130043_sp0.200000.p",
    "lunar_gbox_2_20210905_130047_sp0.200000.p",
    "lunar_gbox_3_20210905_130051_sp0.200000.p",
    "lunar_gbox_4_20210905_130057_sp0.200000.p",
    "lunar_gbox_5_20210905_130101_sp0.200000.p",
    "lunar_gbox_6_20210905_130104_sp0.200000.p",
    "lunar_gbox_7_20210905_130113_sp0.200000.p",
    "lunar_gbox_8_20210905_130116_sp0.200000.p",
]

all_baseline_times = []
all_pifuzz_times = []
for idx, lfile in enumerate(lunar_files):
    r_seed = idx
    game = EW.Wrapper("lunar")
    game.create_environment(env_seed=r_seed)
    game.create_model(agent_path, r_seed)

    pop_summ, pool, tot_warns_mm_e, tot_warns_mm_h, num_cycles, total_trials, total_time, avg_time = pickle.load(open(lfile, "r"))
    bline_warn_times = []
    pifuzz_warn_times = []

    for seed in pool:
        agent_reward, _, _ = game.run_pol_fuzz(seed.data, "qualitative")
        if agent_reward == 0:
            bline_warn_times.append(seed.gen_time)
        if agent_reward == 0 and seed.num_warn_mm_hard > 0:
            pifuzz_warn_times.append(seed.gen_time)

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
plt.savefig("%s_warn_overtime_%d_%s.pdf" % ("lunar", FUZZ_BUDGET, "bline"), bbox_inches="tight")