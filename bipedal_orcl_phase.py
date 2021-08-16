import os
import time
import pickle
import psutil

path_to_pool = ""
_, pool = pickle.load(open(path_to_pool, "rb"))

ptr = 0
gpu_id = 0
while ptr < len(pool):
    cpu_utils = psutil.cpu_percent(percpu=True)
    for cpu_id, cu in enumerate(cpu_utils):
        if cu < 5:
            command = "CUDA_VISIBLE_DEVICES=%d taskset -c %d python run.py -E bipedal -R 1 -A final_policies/bipedal_org -SP %d -PP %s" % (gpu_id, cpu_id, ptr, path_to_pool)
            os.system(command)

            gpu_id += 1
            gpu_id %= 3

            ptr += 1

    time.sleep(60)