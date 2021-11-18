"""
Visualizes training result to replicate plots in paper Deep Reinforcement Learning for Option Replication and Hedging
"""
import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

def get_action_fn_logs(save_dir):
    return pd.read_hdf(os.path.join(save_dir, 'action_fn_logs.h5'))

def get_evaluation_path(save_dir):
    action_eval_path = os.path.join(save_dir, 'eval_path.pkl')
    return pickle.load(open(action_eval_path, 'rb'))

def get_reward_history_plot(oos_eval, cfg):
    """
    At each call back, we keep num of steps per episode * number of out of sample paths rows for the rewards
    we need to compute the episode length so we know the point at which we snapshot the training
    """
    rows_per_callback = cfg['episode_length'] * cfg['num_out_of_sample_path']
    oos_eval['idx'] = np.repeat(np.arange(oos_eval.shape[0] // rows_per_callback), rows_per_callback)
    fig, ax = plt.subplots(figsize=(12, 8))
    reward_history = oos_eval.groupby('idx').reward.median()
    reward_history.plot(style='x-', ax=ax)
    ax.set_title('Average Reward versus Data Size')
    return fig, ax

def get_sample_hedging_path_plot(oos_eval, eval_path, cfg, path_idx=None):
    """
    get a plot of out of sample path and prediction result
    path_idx: the index of the path we want to show, if None, pick a random one
    """
    if path_idx is None:
        path_idx = np.random.randint(cfg['num_out_of_sample_path'])
    eval_path = eval_path.copy()
    eval_path[['holdings', 'reward']] = oos_eval.query('idx==idx.max()')[['holdings', 'reward']].copy()
    sample_experiment = eval_path.query('episode_idx==@path_idx')
    episode_length = cfg['episode_length']
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].plot(np.arange(episode_length), sample_experiment.delta * -100, '-')
    axes[0].plot(np.arange(episode_length), sample_experiment.holdings, '*-')
    axes[0].set_title('Theoretical delta vs Predicted delta')
    axes[0].legend(['Theoretical delta * -100', 'Predicted stock holding'])
    axes[1].plot(np.arange(episode_length), ((eval_path.query('episode_idx==@path_idx')['call_price'].shift(-1) - \
        eval_path.query('episode_idx==@path_idx')['call_price']) * 100).cumsum(), 'x-')
    axes[1].plot(np.arange(episode_length), ((sample_experiment['price'].shift(-1) - \
        sample_experiment['price']) * sample_experiment['holdings']).cumsum(), 'v-')
    axes[1].set_title('Option Pnl vs Hedge Pnl')
    axes[1].legend(['Option Pnl', 'Hedge Pnl'])
    return fig, axes

def make_post_training_plots(save_dir, cfg, sample_path_idx=None):
    eval_path = get_evaluation_path(save_dir)
    oos_eval = get_action_fn_logs(save_dir)
    fig, _ = get_reward_history_plot(oos_eval, cfg)
    fig.savefig(os.path.join(save_dir, 'reward_history.pdf'))
    fig, _ = get_sample_hedging_path_plot(oos_eval, eval_path, cfg, sample_path_idx)
    fig.savefig(os.path.join(save_dir, 'sample_path.pdf'))

# TODO
# add visualization for kde plot
# add visualization for plot around expiration