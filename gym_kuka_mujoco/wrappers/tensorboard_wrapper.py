import numpy as np
import tensorflow as tf
from tensorflow.python.summary.writer.writer import FileWriter
from gym import Wrapper
from stable_baselines.common.vec_env import VecEnvWrapper

class TBWrapper(Wrapper):

    def __init__(self, env, logdir, info_keywords=()):
        """
        A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.
        :param env: (Gym environment) The environment
        :param filename: (str) the location to save tensorboard logs
        :param info_keywords: (tuple) extra information to log, from the information return of environment.step
        """
        Wrapper.__init__(self, env=env)
        self.writer = FileWriter(logdir)
        self.info_keywords = info_keywords
        self.episode_info = dict()
        self.total_steps = 0

    def step(self, action):
        """
        Step the environment with the given action
        :param action: ([int] or [float]) the action
        :return: ([int] or [float], [float], [bool], dict) observation, reward, done, information
        """
        observation, reward, done, info = self.env.step(action)

        for key in self.info_keywords:
            if key not in info:
                break
            if key in self.episode_info:
                self.episode_info[key].append(info[key])
            else:
                self.episode_info[key] = [info[key]]

        if done:
            # Compute data summaries.
            summary_values = []
            for key, value in self.episode_info.items():
                mean = np.mean(value)
                std = np.std(value)
                minimum = np.min(value)
                maximum = np.max(value)
                total = np.sum(value)

                summary_values.extend([
                    tf.Summary.Value(tag="eval/" + key + "/mean", simple_value=mean),
                    tf.Summary.Value(tag="eval/" + key + "/std", simple_value=std),
                    tf.Summary.Value(tag="eval/" + key + "/min", simple_value=minimum),
                    tf.Summary.Value(tag="eval/" + key + "/max", simple_value=maximum),
                    tf.Summary.Value(tag="eval/" + key + "/sum", simple_value=total),
                ])
            summary = tf.Summary(value=summary_values)
            self.writer.add_summary(summary, self.total_steps)
            
            # Clear the episode_info dictionary
            self.episode_info = dict()

        self.total_steps += 1
        return observation, reward, done, info

    def reset(self, **kwargs):
        """
        Invokes the reset method of the underlying environment, passing along any keywords.
        """
        return self.env.reset(**kwargs)

    def close(self):
        """
        Closes the FileWriter and the underlying environment.
        """
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

        self.env.close()

class TBVecEnvWrapper(VecEnvWrapper):

    def __init__(self, venv, logdir, info_keywords=(), **kwargs):
        """
        A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.
        :param env: (Gym environment) The environment
        :param filename: (str) the location to save tensorboard logs
        :param info_keywords: (tuple) extra information to log, from the information return of environment.step
        """
        VecEnvWrapper.__init__(self, venv=venv, **kwargs)
        self.writer = FileWriter(logdir)
        self.info_keywords = info_keywords
        self.episode_infos = [dict() for _ in range(self.venv.num_envs)]
        self.total_steps = 0

    def step_wait(self):
        """
        Step the environment with the given action
        :param action: ([int] or [float]) the action
        :return: ([int] or [float], [float], [bool], dict) observation, reward, done, information
        """
        obs, rews, dones, infos = self.venv.step_wait()

        for i in range(self.venv.num_envs):
            for key in self.info_keywords:
                if key not in infos[i]:
                    break
                if key in self.episode_infos[i]:
                    self.episode_infos[i][key].append(infos[i][key])
                else:
                    self.episode_infos[i][key] = [infos[i][key]]

            if dones[i]:
                # Compute data summaries.
                summary_values = []
                for key, value in self.episode_infos[i].items():
                    mean = np.mean(value)
                    std = np.std(value)
                    minimum = np.min(value)
                    maximum = np.max(value)
                    total = np.sum(value)

                    summary_values.extend([
                        tf.Summary.Value(tag="eval/" + key + "/mean", simple_value=mean),
                        tf.Summary.Value(tag="eval/" + key + "/std", simple_value=std),
                        tf.Summary.Value(tag="eval/" + key + "/min", simple_value=minimum),
                        tf.Summary.Value(tag="eval/" + key + "/max", simple_value=maximum),
                        tf.Summary.Value(tag="eval/" + key + "/sum", simple_value=total),
                    ])
                summary = tf.Summary(value=summary_values)
                self.writer.add_summary(summary, self.total_steps + i)
                
                # Clear the episode_infos dictionary
                self.episode_infos[i] = dict()

        self.total_steps += self.venv.num_envs
        return obs, rews, dones, infos

    def reset(self, **kwargs):
        """
        Invokes the reset method of the underlying environment, passing along any keywords.
        """
        return self.venv.reset(**kwargs)

    def close(self):
        """
        Closes the FileWriter and the underlying environment.
        """
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

        self.env.close()
