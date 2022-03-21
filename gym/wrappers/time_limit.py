from typing import Optional

import gym


class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        self._elapsed_steps += 1

        if truncated:
            assert done, "`truncated` cannot be True when `done` is False"
        elif self._elapsed_steps >= self._max_episode_steps:
            truncated = (
                True if not done else False
            )  # `truncated = False` if episode terminates and truncates on the same step
            done = True
        return observation, reward, done, truncated, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)
