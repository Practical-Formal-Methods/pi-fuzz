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

    # for _ in range(100):
    #     game.env.reset(None)
    #     state, _ = game.env.get_state(one_hot=True, linearize=True, window=True, distance=True)
    #     agent_reward, _, _ = game.run_pol_fuzz(state, mode="quantitative")
    #     print(agent_reward)
    # exit()

    mutator = Mutator.RandomActionMutator(game)
    # schedule = Scheduler.RandomScheduler()  # using random scheduler can be problematic. check pool population code
    schedule = Scheduler.QueueScheduler()
    # oracle = Oracle.LookAheadOracle(game, mode="quantitative")
    oracle = Oracle.MetamorphicOracle(game, mode="qualitative")

    num_tot_warns = []
    num_ind_warns = []
    for _ in range(8):
        fuzzer = Fuzzer.Fuzzer(fuzz_game=game, schedule=schedule, oracle=oracle, mutator=mutator)
        warnings = fuzzer.fuzz()

        ind_warn = 0
        for wrn in warnings:
            if wrn > 0:
                ind_warn += 1
        num_ind_warns.append(ind_warn)
        num_tot_warns.append(sum(warnings))
        print(ind_warn, sum(warnings))

    print(num_ind_warns)
    print(num_tot_warns)
    # bad, meta, quantitative over 69 seeds
    # [44, 32, 34, 37, 35, 34, 36, 37]
    # [758, 538, 641, 683, 721, 736, 600, 721]
    #good, meta, quantitative over 66 seeds
    # [7, 8, 8, 15, 11, 8, 3, 7]
    # [37, 76, 27, 215, 131, 85, 42, 35]
num_wrng = fuzz_func("policies/modagent.pth")