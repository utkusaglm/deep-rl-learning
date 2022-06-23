import gym
from stable_baselines3 import PPO
import os
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

env = gym.make('LunarLander-v2')
env_id = "LunarLander-v2"

models_dir = "models/PPO"
n_envs= 5
logdir = "logs"
eval_envs = make_vec_env(env_id, n_envs=5)
eval_freq = int(1e5)
eval_freq = max(eval_freq // n_envs, 1)

# env = make_vec_env('LunarLander-v2', n_envs=32)
env = gym.make('LunarLander-v2')
model = PPO(
        policy = 'MlpPolicy',
        env = env,
        n_steps = 5000,
        batch_size = 4096,
        learning_rate=0.003,
        clip_range=0.3,
        n_epochs = 30,
        gamma = 0.999,
        vf_coef=0.5,
        gae_lambda = 0.98,
        ent_coef = 0.01,
        verbose=1)

def train_model(timesteps=1000, iter=10, saved_model_fn_int=0):
    #saved_model_fn_int: this parameter is for using past model.
    TIMESTEPS = timesteps
    for i in range(iter):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
        saved_model_file_name = saved_model_fn_int + i * TIMESTEPS
        model.save(f"{models_dir}/{saved_model_file_name}")

if not os.path.exists(logdir):
    os.makedirs(logdir)

train_model(timesteps=1000, iter=30, saved_model_fn_int='PPO')