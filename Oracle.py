import time
import logging
import itertools
import numpy as np
import Mutator
from fuzz_config import ORACLE_SEARCH_BUDGET, BUG_CONFIRMATION_BUDGET
from abc import ABC, abstractmethod
from linetrack.magic_oracle.magic import magic_oracle

logger = logging.getLogger("fuzz_logger")


class Oracle(ABC):

    def __init__(self, game, rand_seed, delta=None, de_dup=None, orcl_mut_bdgt=None):
        # super().__init__()
        self.game = game
        self.mode = 'qualitative'
        self.rand_seed = rand_seed
        self.rng = np.random.default_rng(rand_seed)
        self.delta = delta
        self.de_dup = de_dup

        if game.env_iden == "highway":
            self.mutator = Mutator.LinetrackOracleMutator(game, orcl_mut_bdgt)
        elif game.env_iden == "lunar":
            self.mutator = Mutator.LunarOracleMoonHeightMutator(game)
        elif game.env_iden == "bipedal":
            self.mutator = Mutator.BipedalEasyOracleMutator(game)

    @abstractmethod
    def explore(self, fuzz_seed):
        pass

    def setRandAndFuzzSeed(self, fuzz_seed, rand_seed=None):
        if not rand_seed:
            rand_seed =  self.rand_seed

        if self.game.env_iden == "linetrack":
            idl_rng = np.random.default_rng(rand_seed)
            self.game.env.reset(idl_rng)
        else:
            self.game.env.seed(rand_seed)

        self.game.set_state(fuzz_seed.hi_lvl_state)  # linetrack: [street, v])


class MMBugOracle(Oracle):
    def __init__(self, game, rand_seed):

        if self.game.env_iden == "bipedal":
            print("This oracle is not suitable for BipedalWalker environment as it takes too much time. Thus, we did not include this experiment in the paper. If you are curious to try this, remove this condition and consider decreasing confirmation budget.")
            exit()

        super().__init__(game, rand_seed)

    def explore(self, fuzz_seed):

        time_spent = 0
        s = time.time()

        org_f_cnt = 0
        # below loop corresponds to line 2 in Algorithm 1 
        for rand_seed in range(BUG_CONFIRMATION_BUDGET):  
            self.setRandAndFuzzSeed(fuzz_seed, rand_seed)
            org_rew, _, _ = self.game.run_pol_fuzz(fuzz_seed.data)
            if org_rew == 0: org_f_cnt += 1

        for _ in range(ORACLE_SEARCH_BUDGET):
            mut_state = self.mutator.mutate(fuzz_seed, self.rng, mode='unrelax')
            if mut_state is None:
                num_rejects += 1
                continue

            mut_f_cnt = 0
            # below loop corresponds to line 5 in Algorithm 1 
            for rand_seed in range(BUG_CONFIRMATION_BUDGET):                
                self.setRandAndFuzzSeed(fuzz_seed, rand_seed)
                nn_state, _ = self.game.get_state()
                mut_rew, _, _ = self.game.run_pol_fuzz(nn_state)
                
                if mut_rew == 0: mut_f_cnt += 1
            
            # below condition effectively corresponds to line 6 in Algorithm 1
            if mut_f_cnt < org_f_cnt:
                e = time.time()
                time_spent += e-s
                return 1, time_spent  # bug found
        
        e = time.time()
        time_spent += e-s
        return 0, time_spent  # bug not found


class MMSeedBugBasicOracle(Oracle):
    def __init__(self, game, rand_seed):
        super().__init__(game, rand_seed)
    
    def explore(self, fuzz_seed):

        s = time.time()

        self.setRandAndFuzzSeed(fuzz_seed)
        org_reward, _, _ = self.game.run_pol_fuzz(fuzz_seed.data)

        if org_reward == 0: fuzz_seed.is_crash = True

        for _ in range(ORACLE_SEARCH_BUDGET):

            mut_state = self.mutator.mutate(fuzz_seed, self.rng, mode='unrelax')
            if mut_state is None:
                num_rejects += 1
                continue

            self.setRandAndFuzzSeed(mut_state)

            nn_state, _ = self.game.get_state()

            mut_reward, _, _ = self.game.run_pol_fuzz(nn_state)

            if mut_reward - org_reward > self.delta:
                e = time.time()
                time_spent = e-s
                return 1, time_spent
        
        time_spent = e-s
        return 0, time_spent

class MMSeedBugExtOracle(Oracle):
    def __init__(self, game, rand_seed):
        super().__init__(game, rand_seed)

    def explore(self, fuzz_seed):
        s = time.time()

        self.setRandAndFuzzSeed(fuzz_seed)

        org_reward, _, _ = self.game.run_pol_fuzz(fuzz_seed.data)

        if org_reward == 0: fuzz_seed.is_crash = True

        num_bugs = 0
        for _ in range(ORACLE_SEARCH_BUDGET):
            mut_state = self.mutator.mutate(fuzz_seed, self.rng, mode='relax')
            if mut_state is None:
                num_rejects += 1
                continue

            self.setRandAndFuzzSeed(mut_state)

            nn_state, _ = self.game.get_state()

            mut_reward, _, _ = self.game.run_pol_fuzz(nn_state)

            if mut_reward - org_reward > self.delta:
                num_bugs += 1
        
        e = time.time()
        time_spent = e-s
        return num_bugs, time_spent

class MMSeedBug2BugOracle(Oracle):

    def __init__(self, game, rand_seed):
        super().__init__(game, rand_seed)
        
    def explore(self, fuzz_seed):
        s = time.time()
        
        mmseedbugoracle = MMSeedBugBasicOracle(Oracle)
        is_seed_bug, _ = mmseedbugoracle.explore(fuzz_seed)
        
        if is_seed_bug == 1:
            mmbugoracle = MMBugOracle(Oracle)
            is_bug, _ = mmbugoracle.explore(fuzz_seed)
            e = time.time()
            time_spent = e-s
            if is_bug == 1: return 1, time_spent

        e = time.time()
        time_spent = e-s
        return 0, time_spent


class FailureSeedBugOracle(Oracle):
    def __init__(self, game, rand_seed):
        super().__init__(game, rand_seed)
        
    def explore(self, fuzz_seed):
        s = time.time()
        
        self.setRandAndFuzzSeed(fuzz_seed)

        org_reward, _, _ = self.game.run_pol_fuzz(fuzz_seed.data)
        
        e = time.time()
        time_spent = e-s
        
        if org_reward == 0: 
            fuzz_seed.is_crash = True
            return 1, time_spent
        return 0, time_spent


class RuleSeedBug(Oracle):
    def __init__(self, game, rand_seed):
        super().__init__(game, rand_seed)
        
    def explore(self, fuzz_seed):
        s = time.time()
        
        self.setRandAndFuzzSeed(fuzz_seed)

        org_reward, org_play, _ = self.game.run_pol_fuzz(fuzz_seed.data)
        
        for _ in range(ORACLE_SEARCH_BUDGET):
            mut_state = self.mutator.mutate(fuzz_seed, self.rng, mode='unrelax')
            if mut_state is None:
                num_rejects += 1
                continue

            self.setRandAndFuzzSeed(mut_state)

            nn_state, _ = self.game.get_state()

            mut_reward, mut_play, _ = self.game.run_pol_fuzz(nn_state)
            
            # should be winning the game. we return only true positives for this oracle
            if mut_reward == 100 and mut_reward > org_reward:
                e = time.time()
                time_spent = e-s
                # rule: if there is policy that can succeed in harder state, it should take the same action on the easier state
                if self.game.env_iden == "bipedal":
                    if list(mut_play[0]) != list(org_play[0]): 
                        return 1, time_spent
                else:
                    if mut_play[0] != org_play[0]: 
                        return 1, time_spent
        
        e = time.time()
        time_spent = e-s
        
        return 0, time_spent


class PerfectBugOracle(Oracle):

    def __init__(self, game, rand_seed):
        if not self.game.env_iden == "highway":
            print("This oracle is only suitable for Highway!")
            exit()
        super().__init__(game, rand_seed)

    def explore(self, fuzz_seed):
        s = time.time()


        e = time.time()
        time_spent = e-s
        
        return 0, time_spent

class PerfectSeedBugOracle(Oracle):

    def __init__(self, game, rand_seed):
        if not self.game.env_iden == "highway":
            print("This oracle is only suitable for Highway!")
            exit()
        super().__init__(game, rand_seed)
        
    def explore(self, fuzz_seed):
        s = time.time()
        self.setRandAndFuzzSeed(fuzz_seed)

        org_reward, _, _ = self.game.run_pol_fuzz(fuzz_seed.data)
        if org_reward <= 0:
            self.setRandAndFuzzSeed(fuzz_seed)
            if magic_oracle(self.game.env):
                e = time.time()
                time_spent = e-s

                return 1, time_spent

        e = time.time()
        time_spent = e-s
        
        return 0, time_spent


class PerfectBugOracle(Oracle):
    def __init__(self, game, rand_seed):
        if not self.game.env_iden == "highway":
            print("This oracle is only suitable for Highway!")
            exit()
        super().__init__(game, rand_seed)
    
    def explore(self, fuzz_seed):
        s = time.time()

        num_perfect_fails = 0
        for rs in range(BUG_CONFIRMATION_BUDGET):
            self.setRandAndFuzzSeed(fuzz_seed, rs)
            if not magic_oracle(self.game.env): num_perfect_fails += 1

        num_policy_fails = 0
        for rs in range(BUG_CONFIRMATION_BUDGET):
            self.setRandAndFuzzSeed(fuzz_seed, rs)
            rew, _, _ = self.game.run_pol_fuzz(fuzz_seed.data, self.mode)
            if rew == 0: num_policy_fails += 1

        e = time.time()
        time_spent = e-s
        
        if num_policy_fails > num_perfect_fails:
            return 1, time_spent
        
        return 0, time_spent
