import time
import Oracle
import Scheduler
import EnvWrapper as EW
import Fuzzer
import Mutator

def fuzz_func(agent_path):

    game = EW.Wrapper(agent_path)
    game.create_linetrack_environment()
    game.create_linetrack_model()

    mutator = Mutator.RandomActionMutator()
    # schedule = Scheduler.RandomScheduler()  # using random scheduler can be problematic. check pool population code
    schedule = Scheduler.QueueScheduler()
    # oracle = Oracle.LookAheadOracle(game)
    oracle = Oracle.MetamorphicOracle(game, mode="quantitative")

    fuzzer = Fuzzer.Fuzzer(fuzz_game=game, schedule=schedule, oracle=oracle, mutator=mutator)

    s = time.time()
    fuzzer.fuzz()
    e = time.time()
    print(e - s)

    return fuzzer.warning_cnt

num_wrng = fuzz_func("policies/modagent.pth")