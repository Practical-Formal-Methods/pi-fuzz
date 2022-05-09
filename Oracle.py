import copy
import logging
import numpy as np
import Mutator
from fuzz_config import ORACLE_SEARCH_BUDGET, BUG_CONFIRMATION_BUDGET
from abc import ABC, abstractmethod

logger = logging.getLogger("fuzz_logger")


class Oracle(ABC):

    def __init__(self, game, rand_seed, de_dup=None):
        self.game = game
        self.mode = 'qualitative'
        self.rng = np.random.default_rng(rand_seed)
        self.de_dup = de_dup

        if game.env_iden == "highway":
            orcl_mut_bdgt = 2  # remove or add 2 cars
            self.mutator = Mutator.HighwayOracleMutator(game, orcl_mut_bdgt)
        elif game.env_iden == "lunar":
            self.mutator = Mutator.LunarOracleMoonHeightMutator(game)
        elif game.env_iden == "bipedal":
            self.mutator = Mutator.BipedalEasyOracleMutator(game)

    @abstractmethod
    def explore(self, fuzz_seed):
        pass

    def setRandAndFuzzSeed(self, env_state, rand_state=None, rand_seed=None):
        # if not rand_seed:
        #     rand_seed =  self.rand_seed

        # random state is either restored to original or resetted with rand_seed
        if rand_seed is not None:
            if self.game.env_iden == "highway":
                idl_rng = np.random.default_rng(rand_seed)
                self.game.env.reset(idl_rng)
            else:
                self.game.env.seed(rand_seed)

            self.game.set_state(env_state)  # highway: [street, v])
        else:
            self.game.set_state(env_state, rand_state)


class MMBugOracle(Oracle):
    def __init__(self, game, rand_seed):
        super().__init__(game, rand_seed)

    def explore(self, fuzz_seed):
        if self.game.env_iden == "bipedal":
            print("This oracle is not suitable for BipedalWalker environment as it takes too much time. Thus, we did not include this experiment in the paper. If you are curious to try this, remove this condition and consider decreasing confirmation budget.")
            exit()

        org_f_cnt = 0
        # below loop corresponds to line 2 in Algorithm 1 
        for rand_seed in range(BUG_CONFIRMATION_BUDGET):  
            self.setRandAndFuzzSeed(fuzz_seed.hi_lvl_state, rand_seed=rand_seed)
            org_rew, _, _ = self.game.play(fuzz_seed.data)
            if org_rew == 0: org_f_cnt += 1

        num_rejects = 0
        for _ in range(ORACLE_SEARCH_BUDGET):
            mut_state = self.mutator.mutate(fuzz_seed, self.rng, mode='unrelax')
            if mut_state is None:
                num_rejects += 1
                continue

            mut_f_cnt = 0
            # below loop corresponds to line 5 in Algorithm 1 
            for rand_seed in range(BUG_CONFIRMATION_BUDGET):                
                self.setRandAndFuzzSeed(fuzz_seed.hi_lvl_state, rand_seed=rand_seed)
                nn_state, _ = self.game.get_state()
                mut_rew, _, _ = self.game.play(nn_state)
                
                if mut_rew == 0: mut_f_cnt += 1
            
            # below condition effectively corresponds to line 6 in Algorithm 1
            if mut_f_cnt < org_f_cnt:
                logger.info("%d out of %d (un)relaxation is rejected on fuzz seed %d." % (num_rejects, ORACLE_SEARCH_BUDGET, fuzz_seed.identifier))
                return 1  # bug found
        
        logger.info("%d out of %d (un)relaxation is rejected on fuzz seed %d." % (num_rejects, ORACLE_SEARCH_BUDGET, fuzz_seed.identifier))
        return 0  # bug not found


class MMSeedBugBasicOracle(Oracle):
    def __init__(self, game, rand_seed):
        super().__init__(game, rand_seed)
    
    def explore(self, fuzz_seed):

        self.setRandAndFuzzSeed(fuzz_seed.hi_lvl_state, rand_state=fuzz_seed.rand_state)
        org_reward, _, _ = self.game.play(fuzz_seed.data)

        if org_reward == 0: fuzz_seed.is_crash = True
        elif org_reward == 100: 
            logger.info("Skipping fuzz seed %d as agent wins on there (easier state).")
            return 0

        num_rejects = 0
        for _ in range(ORACLE_SEARCH_BUDGET):
            mut_state = self.mutator.mutate(fuzz_seed, self.rng, mode='unrelax')
            if mut_state is None:
                num_rejects += 1
                continue

            self.setRandAndFuzzSeed(mut_state, rand_state=fuzz_seed.rand_state)

            nn_state, _, _ = self.game.get_state()

            mut_reward, _, _ = self.game.play(nn_state)

            if mut_reward > org_reward:
                logger.info("%d out of %d (un)relaxation is rejected on fuzz seed %d." % (num_rejects, ORACLE_SEARCH_BUDGET, fuzz_seed.identifier))
                return 1

        logger.info("%d out of %d (un)relaxation is rejected on fuzz seed %d." % (num_rejects, ORACLE_SEARCH_BUDGET, fuzz_seed.identifier))
        return 0

class MMSeedBugExtOracle(Oracle):
    def __init__(self, game, rand_seed):
        super().__init__(game, rand_seed)

    def explore(self, fuzz_seed):
        self.setRandAndFuzzSeed(fuzz_seed.hi_lvl_state, rand_state=fuzz_seed.rand_state)

        org_reward, _, _ = self.game.play(fuzz_seed.data)

        if org_reward == 0: fuzz_seed.is_crash = True

        num_bugs = 0
        num_rejects = 0
        for _ in range(ORACLE_SEARCH_BUDGET):
            mut_state = self.mutator.mutate(fuzz_seed, self.rng, mode='relax')
            if mut_state is None:
                num_rejects += 1
                continue

            self.setRandAndFuzzSeed(mut_state, rand_state=fuzz_seed.rand_state)

            nn_state, _, _ = self.game.get_state()

            mut_reward, _, _ = self.game.play(nn_state)

            if mut_reward > org_reward:
                num_bugs += 1
        
        logger.info("%d out of %d (un)relaxation is rejected on fuzz seed %d." % (num_rejects, ORACLE_SEARCH_BUDGET, fuzz_seed.identifier))

        return num_bugs


class MMSeedBug2BugOracle(Oracle):

    def __init__(self, game, rand_seed):
        super().__init__(game, rand_seed)
        
    def explore(self, fuzz_seed):        
        mmseedbugoracle = MMSeedBugBasicOracle(Oracle)
        is_seed_bug, _ = mmseedbugoracle.explore(fuzz_seed)
        
        if is_seed_bug == 1:
            mmbugoracle = MMBugOracle(Oracle)
            is_bug, _ = mmbugoracle.explore(fuzz_seed)
            if is_bug == 1: return 1

        return 0


class FailureSeedBugOracle(Oracle):
    def __init__(self, game, rand_seed):
        super().__init__(game, rand_seed)
        
    def explore(self, fuzz_seed):
        
        self.setRandAndFuzzSeed(fuzz_seed.hi_lvl_state, fuzz_seed.rand_state)

        org_reward, _, _ = self.game.play(fuzz_seed.data)
        
        if org_reward == 0: 
            fuzz_seed.is_crash = True
            return 1

        return 0


class RuleSeedBugOracle(Oracle):
    def __init__(self, game, rand_seed):
        super().__init__(game, rand_seed)
        
    def explore(self, fuzz_seed):
        
        self.setRandAndFuzzSeed(fuzz_seed.hi_lvl_state, fuzz_seed.rand_state)

        org_reward, org_play, _ = self.game.play(fuzz_seed.data)
        
        num_rejects = 0
        for _ in range(ORACLE_SEARCH_BUDGET):
            mut_state = self.mutator.mutate(fuzz_seed, self.rng, mode='unrelax')
            if mut_state is None:
                num_rejects += 1
                continue

            self.setRandAndFuzzSeed(mut_state, rand_state=fuzz_seed.rand_state)

            nn_state, _, _ = self.game.get_state()

            mut_reward, mut_play, _ = self.game.play(nn_state)
            
            # should be winning the game. we return only true positives for this oracle
            if mut_reward == 100 and mut_reward > org_reward:
                # rule: if there is policy that can succeed in harder state, it should take the same action on the easier state
                if self.game.env_iden == "bipedal":
                    if list(mut_play[0]) != list(org_play[0]): 
                        logger.info("%d out of %d (un)relaxation is rejected on fuzz seed %d." % (num_rejects, ORACLE_SEARCH_BUDGET, fuzz_seed.identifier))
                        return 1
                else:
                    if mut_play[0] != org_play[0]: 
                        logger.info("%d out of %d (un)relaxation is rejected on fuzz seed %d." % (num_rejects, ORACLE_SEARCH_BUDGET, fuzz_seed.identifier)) 
                        return 1
        
        logger.info("%d out of %d (un)relaxation is rejected on fuzz seed %d." % (num_rejects, ORACLE_SEARCH_BUDGET, fuzz_seed.identifier))
        return 0


class PerfectOracle(Oracle):
    def magic_oracle(env):
        envs = [env]

        while len(envs) != 0:
            env = envs.pop()
            for action in env.applicable_actions():
                current_env = copy.deepcopy(env)
                reward, _, done = current_env.step(action)
                if done:
                    if reward > 0:
                        return True
                    else:
                        continue
                envs.append(current_env)
        return False


class PerfectSeedBugOracle(PerfectOracle):
    def __init__(self, game, rand_seed):
        super().__init__(game, rand_seed)
        
    def explore(self, fuzz_seed):
        if not self.game.env_iden == "highway":
            print("This oracle is only suitable for Highway!")
            exit()

        self.setRandAndFuzzSeed(fuzz_seed.hi_lvl_state, fuzz_seed.rand_state)

        org_reward, _, _ = self.game.play(fuzz_seed.data)
        # if the agent is already winning then no need for exploration
        if org_reward <= 0:
            self.setRandAndFuzzSeed(fuzz_seed.hi_lvl_state, fuzz_seed.rand_state)
            if self.magic_oracle(self.game.env):
                return 1
        
        return 0


class PerfectBugOracle(PerfectOracle):
    def __init__(self, game, rand_seed):
        super().__init__(game, rand_seed)

    def explore(self, fuzz_seed):
        if not self.game.env_iden == "highway":
            print("This oracle is only suitable for Highway!")
            exit()

        num_perfect_fails = 0
        for rs in range(BUG_CONFIRMATION_BUDGET):
            self.setRandAndFuzzSeed(fuzz_seed.hi_lvl_state, random_seed=rs)
            if not self.magic_oracle(self.game.env): num_perfect_fails += 1

        num_policy_fails = 0
        for rs in range(BUG_CONFIRMATION_BUDGET):
            self.setRandAndFuzzSeed(fuzz_seed.hi_lvl_state, random_seed=rs)
            rew, _, _ = self.game.play(fuzz_seed.data, self.mode)
            if rew == 0: num_policy_fails += 1
        
        if num_policy_fails > num_perfect_fails:
            return 1
        
        return 0
