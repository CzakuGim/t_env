from gym.envs.registration import register

register(
    id='trading_env-v0',
    entry_point='trading_env.envs:TradingEnv',
)