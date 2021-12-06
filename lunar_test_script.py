import EnvWrapper as EW

game = EW.Wrapper("lunar")
game.create_environment(env_seed=123)
game.create_model("policies/lunar_org", 123)

game.env.reset()
nn_state, hi_lvl_state = game.get_state()

# when you want to set the environment to a particular state, you can use this function. 
# nn_state is agent's obervation in the current state, and hi_lvl_state contains all the necessary 
# information to create the environment at the particular state
# game.set_state(hi_lvl_state)

reward, actions, visited_states = game.run_pol_fuzz(nn_state)
print(reward, len(actions))

