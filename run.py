import time
import Oracle
import Scheduler
import EnvWrapper as EW
import Fuzzer

from tqdm import trange


def fuzz_func(agent_path, fuzz_type="black"):

    game = EW.Wrapper(agent_path)
    game.create_linetrack_environment()
    game.create_linetrack_model()

    schedule = Scheduler.RandomScheduler()
    # oracle = Oracle.LookAheadOracle(game)
    oracle = Oracle.MetamorphicOracle(game)

    fuzzer = Fuzzer.Fuzzer(fuzz_game=game, fuzz_type=fuzz_type, schedule=schedule, oracle=oracle)

    s = time.time()
    fuzzer.fuzz()
    e = time.time()
    print(e - s)

    return fuzzer.warning_cnt

num_wrng = fuzz_func("policies/modagent.pth")