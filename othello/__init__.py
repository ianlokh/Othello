from gymnasium.envs.registration import register

register(
    id='othello-v0',
    entry_point='othello.envs:OthelloEnv',
)
