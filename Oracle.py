import time
import logging
import itertools
import numpy as np
import Mutator
from fuzz_config import ORACLE_SEARCH_BUDGET
from abc import ABC, abstractmethod
from linetrack.magic_oracle.magic import magic_oracle

logger = logging.getLogger("fuzz_logger")


class Oracle(ABC):

    def __init__(self, game, mode, r_seed, delta, de_dup):
        super().__init__()
        self.game = game
        self.mode = mode
        self.r_seed = r_seed
        self.rng = np.random.default_rng(r_seed)
        self.delta = delta
        self.de_dup = de_dup

    def set_deviations(self):
        deviations = list(itertools.product(self.game.action_space, repeat=0))

        if len(deviations) > SEARCH_BUDGET:
            deviations = self.rng.choice(deviations, SEARCH_BUDGET, replace=False)

        self.deviations = deviations

    @abstractmethod
    def explore(self, fuzz_seed):
        pass


class MetamorphicOracle(Oracle):
    def __init__(self, game, mode, r_seed, delta=None, orcl_mut_bdgt=None, de_dup=False):
        super().__init__(game, mode, r_seed, delta, de_dup)
        if game.env_iden == "linetrack":
            self.mutator = Mutator.LinetrackOracleMutator(game, orcl_mut_bdgt)
        elif game.env_iden == "racetrack":
            self.mutator = Mutator.RacetrackOracleWallMutator(game, orcl_mut_bdgt)
        elif game.env_iden == "lunar":
            self.mutator = Mutator.LunarOracleMoonHeightMutator(game)
        elif game.env_iden == "bipedal":
            self.mutator = Mutator.BipedalEasyOracleMutator(game)

    def explore(self, fuzz_seed):
        fail_s = time.time()
        self.game.set_state(fuzz_seed.hi_lvl_state)  # [fuzz_seed.state_env, fuzz_seed.data[-1]])
        agent_reward, org_play, _ = self.game.run_pol_fuzz(fuzz_seed.data, self.mode)
        fail_e = time.time()
        
        if agent_reward == 0: fuzz_seed.is_crash = True

        num_rejects = 0
        num_warning_easy = 0
        num_warning_hard = 0
        num_warning_optimal = 0
        num_warning_fail = 0
        num_warning_rule = 0
        num_warning_ideal = 0
        num_warn_comm = 0  # common warnings with hard and ideal
        num_dupl = 0
        bug_states = []
        mm_ext_time = 0
        mm_base_time = 0
        fail_time = 0
        opt_time = 0
        ideal_time = 0
        for idx in range(ORACLE_SEARCH_BUDGET):
            if self.game.env_iden == "linetrack":
                exp_rng = np.random.default_rng(self.r_seed)
                self.game.env.reset(exp_rng)
            elif self.game.env_iden == "racetrack":
                self.game.env.reset()  # random seed is refreshed inside of reset function
            else:
                self.game.env.seed(self.r_seed)
            # make map EASIER
            if False:  # agent_reward > 0:   # FIX THIS LATER  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                s = time.time()
                mut_state = self.mutator.mutate(fuzz_seed, self.rng, mode='easy')

                if mut_state is None:
                    num_rejects += 1
                    continue
                
                self.game.set_state(mut_state)  # linetrack: [street, v])
                nn_state, hi_state = self.game.get_state()

                if self.de_dup and hi_state in bug_states: 
                    num_dupl += 1
                    continue

                mut_reward, mut_play, _ = self.game.run_pol_fuzz(nn_state, self.mode)
                
                if agent_reward - mut_reward > self.delta:
                    num_warning_easy += 1
                    bug_states.append(hi_state)
                e = time.time()
                mm_ext_time += e-s
            # make map HARDER
            else:
                fail_time = fail_e - fail_s  # calculated once above
                s = time.time()
                num_warning_fail = 1  # fail oracle always returns a bug if the agent fails
                if num_warning_ideal > 0 or num_rejects > 0: continue  # can find only one hard warning on one state   num_warning_hard > 0 and !!!!!!!!!FIX LATER!!!!!!!!!!! 

                mut_state = self.mutator.mutate(fuzz_seed, self.rng, mode='hard')
                if mut_state is None:
                    num_rejects += 1
                    continue

                if not self.game.env_iden == "bipedal":
                    s = time.time()
                    if self.ideal_bug_conf("hard", fuzz_seed, mut_state): num_warning_ideal = 1
                    e = time.time()
                    ideal_time += e-s

                continue  ## !!!!!!!!!!!!!!! FIX LATER

                self.game.set_state(mut_state)  # linetrack: [street, v])
                nn_state, hi_state = self.game.get_state()

                mut_reward, mut_play, _ = self.game.run_pol_fuzz(nn_state, mode=self.mode)
                e = time.time()
                mm_ext_time += e-s
                mm_base_time += e-s

                if mut_reward - agent_reward > self.delta:
                    num_warning_hard = 1   # in this case there can only be one bug which is on the original state
                    if self.game.env_iden == "bipedal":
                        if list(mut_play[0]) != list(org_play[0]): num_warning_rule = 1  # rule: if there is policy that can succeed in harder state,it should take the same action on the easier state
                    else:
                        if mut_play[0] != org_play[0]: num_warning_rule = 1  # rule: if there is policy that can succeed in harder state,it should take the same action on the easier state
                
    
        '''
        if self.game.env_iden == "linetrack":
            s = time.time()
            # note that metam. base oracle is a subset of optimal oracle and it differs from optimal oracle only in this case. 
            # so we run optimal oracle only in below case to save from running time. 
            # optimal oracle returns all bugs that metam oracle found and plus this case. check run.py
            if agent_reward <= 0 and num_warning_hard == 0:  # if we could not found a bug in hard case
                self.game.set_state(fuzz_seed.hi_lvl_state) 
                if magic_oracle(self.game.env):
                    num_warning_optimal = 1
            e = time.time()
            opt_time += e-s
        '''

        time_data = [mm_base_time, mm_ext_time, ideal_time, opt_time, fail_time]

        return num_warning_easy, num_warning_hard, num_warning_optimal, num_warning_fail, num_warning_rule, num_dupl, num_rejects, num_warning_ideal, time_data


    def perfect_ideal_bug(self, fuzz_seed):
        BDGT = 50
        
        num_perf_fails = 0
        for rs in range(BDGT):
            idl_rng = np.random.default_rng(rs)
            self.game.env.reset(idl_rng)
            self.game.set_state(fuzz_seed.hi_lvl_state)  

            if not magic_oracle(self.game.env): num_perf_fails += 1

        actual_ratio = num_perf_fails / BDGT

        num_conc_fails = 0
        for rs in range(BDGT):
            idl_rng = np.random.default_rng(rs)
            self.game.env.reset(idl_rng)
            self.game.set_state(fuzz_seed.hi_lvl_state)  

            idl_org_rew, _, _ = self.game.run_pol_fuzz(fuzz_seed.data, self.mode)
            if idl_org_rew == 0: num_conc_fails += 1

        conc_ratio = num_conc_fails / BDGT

        return conc_ratio > actual_ratio
        

    def ideal_bug_conf(self, rlx_mode, fuzz_seed, mut_state):
        org_f_cnt = 0
        mut_f_cnt = 0
        for rs in range(5):
            if self.game.env_iden == "linetrack":
                idl_rng = np.random.default_rng(rs)
                self.game.env.reset(idl_rng)
            else:
                self.game.env.seed(rs)

            self.game.set_state(fuzz_seed.hi_lvl_state)  
            idl_org_rew, _, _ = self.game.run_pol_fuzz(fuzz_seed.data, self.mode)
            
            if self.game.env_iden == "linetrack":
                idl_rng = np.random.default_rng(rs)
                self.game.env.reset(idl_rng)
            else:
                self.game.env.seed(rs)
            self.game.set_state(mut_state)  # linetrack: [street, v])
            nn_state, hi_state = self.game.get_state()
            idl_mut_rew, _, _ = self.game.run_pol_fuzz(nn_state, mode=self.mode)
            
            if idl_org_rew == 0: org_f_cnt += 1
            if idl_mut_rew == 0: mut_f_cnt += 1

        if rlx_mode == "easy": cond = mut_f_cnt > org_f_cnt 
        else: cond = mut_f_cnt < org_f_cnt 
        
        if cond: return True

        return False


class MetamorphicOracleNumEnv(Oracle):
    def __init__(self, game, mode, r_seed, delta=None, orcl_mut_bdgt=None, de_dup=False):
        super().__init__(game, mode, r_seed, delta, de_dup)
        if game.env_iden == "linetrack":
            self.mutator = Mutator.LinetrackOracleMutator(game, orcl_mut_bdgt)
        elif game.env_iden == "lunar":
            self.mutator = Mutator.LunarOracleMoonHeightMutator(game)
        elif game.env_iden == "bipedal":
            self.mutator = Mutator.BipedalEasyOracleMutator(game)

    def explore(self, fuzz_seed):
        self.game.set_state(fuzz_seed.hi_lvl_state)  # [fuzz_seed.state_env, fuzz_seed.data[-1]])
        agent_reward, org_play, _ = self.game.run_pol_fuzz(fuzz_seed.data, self.mode)
        
        if agent_reward == 0: fuzz_seed.is_crash = True
        else: return 0, 0, 0    ####### REMOVE LATER !!!!!!!! #########


        num_rejects = 0
        num_warning_easy = 0
        num_warning_hard = 0
        bug_states = []
        mm_base_time = 0
        for idx in range(ORACLE_SEARCH_BUDGET):
            if self.game.env_iden == "linetrack":
                exp_rng = np.random.default_rng(self.r_seed)
                self.game.env.reset(exp_rng)
            elif self.game.env_iden == "racetrack":
                self.game.env.reset()  # random seed is refreshed inside of reset function
            else:
                self.game.env.seed(self.r_seed)
            # make map EASIER
            if agent_reward > 0:
                mut_state = self.mutator.mutate(fuzz_seed, self.rng, mode='easy')

                if mut_state is None:
                    num_rejects += 1
                    continue
                
                self.game.set_state(mut_state)  # linetrack: [street, v])
                nn_state, hi_state = self.game.get_state()

                if self.de_dup and hi_state in bug_states: 
                    num_dupl += 1
                    continue

                mut_reward, mut_play, _ = self.game.run_pol_fuzz(nn_state, self.mode)
                
                if agent_reward - mut_reward > self.delta:
                    num_warning_easy += 1
                    bug_states.append(hi_state)
            # make map HARDER
            else:
                if num_warning_hard > 0: continue  # can find only one hard warning on one state

                s = time.time()
                mut_state = self.mutator.mutate(fuzz_seed, self.rng, mode='hard')
                if mut_state is None:
                    num_rejects += 1
                    continue

                self.game.set_state(mut_state)  # linetrack: [street, v])
                nn_state, hi_state = self.game.get_state()

                mut_reward, mut_play, _ = self.game.run_pol_fuzz(nn_state, mode=self.mode)
                e = time.time()
                mm_base_time += e-s
                if mut_reward - agent_reward > self.delta:
                    num_warning_hard = 1   # in this case there can only be one bug which is on the original state


        return num_warning_easy, num_warning_hard, mm_base_time



class OptimalOracle(Oracle):
    def explore(self, seed):
        pass

###################
# # LEGACY CODE # #
###################
class LookAheadOracle(Oracle):
    def __init__(self, game, mode, rng, delta=None, de_dup=False):
        super().__init__(game, mode, rng, de_dup, delta)

    def explore(self, fuzz_seed):
        super().set_deviations()
        self.game.env.reset(rng=self.rng)
        num_warning = 0
        self.game.env.set_state(fuzz_seed)
        agent_reward, _, fp = self.game.run_pol_fuzz(fuzz_seed.data, mode=self.mode)
        # if agent does not crash originally, nothing to do in this mode
        if self.mode == "qualitative" and agent_reward > 0:
            return num_warning  # iow 0

        for deviation in self.deviations:
            self.game.env.set_state(fuzz_seed)
            dev_reward, _, fp = self.game.run_pol_fuzz(fuzz_seed.data, lahead_seq=deviation, mode=self.mode)

            if dev_reward - agent_reward > self.delta:
                num_warning += 1

        return num_warning, 0
