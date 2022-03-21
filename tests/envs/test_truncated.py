import gym
import pytest
from gym.spaces import Discrete
from gym.wrappers import TimeLimit
from gym.vector import AsyncVectorEnv, SyncVectorEnv


class TruncatedTestEnv(gym.Env):
    def __init__(self):
        self.action_space = Discrete(2)
        self.observation_space = Discrete(2)
        self.terminal_timestep = 20

        self.timestep = 0

    def step(self, action):
        self.timestep += 1
        done = True if self.timestep >= self.terminal_timestep else False
        truncated = False

        return 0, 0, done, truncated, {}

    def reset(self):
        self.timestep = 0
        return 0


@pytest.mark.parametrize("time_limit", [10, 20, 30])
def test_truncated(time_limit):
    test_env = TimeLimit(TruncatedTestEnv(), time_limit)

    done = False
    test_env.reset()
    while not done:
        _, _, done, truncated, _ = test_env.step(0)
        assert done or not truncated

    if test_env.terminal_timestep < time_limit:
        assert (
            not truncated
        ), "`truncated` should not be True before time limit is reached"
    elif test_env.terminal_timestep == time_limit:
        assert (
            not truncated
        ), "`truncated` should not be True when termination also occurs at same step "
        # termination overrides truncation
    else:
        assert (
            truncated
        ), "`truncated` should be True when time limit is reached before termination"


def test_truncated_vector():
    env0 = TimeLimit(TruncatedTestEnv(), 10)
    env1 = TimeLimit(TruncatedTestEnv(), 20)
    env2 = TimeLimit(TruncatedTestEnv(), 30)

    async_env = AsyncVectorEnv([lambda: env0, lambda: env1, lambda: env2])
    async_env.reset()
    dones = [False, False, False]
    counter = 0
    while not all(dones):
        counter += 1
        _, _, dones, truncateds, _ = async_env.step(async_env.action_space.sample())
        print(dones, truncateds)
    assert counter == 20
    assert all(truncateds == [True, False, False])

    sync_env = SyncVectorEnv([lambda: env0, lambda: env1, lambda: env2])
    sync_env.reset()
    dones = [False, False, False]
    counter = 0
    while not all(dones):
        counter += 1
        _, _, dones, truncateds, _ = async_env.step(async_env.action_space.sample())
        print(dones, truncateds)
    assert counter == 20
    assert all(truncateds == [True, False, False])
