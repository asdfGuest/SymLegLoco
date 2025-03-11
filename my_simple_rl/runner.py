from simple_rl.utils.utils import time_to_str
from simple_rl.utils.log_tree import LogTree
from simple_rl.runner.utils import EpisodeTracker, StepTimer

from my_simple_rl.env import BaseEnv
from my_simple_rl.ppo import PPO

import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class PPORunner:
    def __init__(
            self,
            env:BaseEnv,
            algo:PPO,
            run_dir:str='runs',
            log_interval:int=1,
            checkpoint_interval:int|None=None,
        ) :
        self.env = env
        self.algo = algo
        self.n_rollout = self.algo.cfg.n_rollout
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        self.acc_iters = 0

        # directories
        self.run_dir = os.path.join(
            run_dir,
            datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        )
        self.checkpoint_dir = os.path.join(self.run_dir, 'checkpoints')

        self.writer = SummaryWriter(log_dir=self.run_dir)
    

    def _save_log(self, log_str:str) :
        os.makedirs(self.run_dir, exist_ok=True)
        log_path = os.path.join(self.run_dir, 'log.txt')
        with open(log_path, 'a', encoding='utf-8') as file:
            file.write(log_str)
    

    def _save_checkpoint(self) :
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.algo.save(
            os.path.join(self.checkpoint_dir, 'model_%d.pt'%self.acc_iters)
        )
    

    def train(self, train_iters:int) :
        episode_tracker = EpisodeTracker(self.env.spec)
        step_timer = StepTimer(train_iters * self.n_rollout)
        logger = LogTree()

        internal_module = self.algo.internal_module

        obs_pack, env_info = self.env.reset()
        state = internal_module.compute_state(obs_pack)

        for i_iter in range(1, train_iters + 1) :
            self.acc_iters += 1

            for _ in range(self.n_rollout) :
                action = self.algo.act(state)
                next_obs_pack, reward, terminated, truncated, env_info = self.env.step(action)
                next_state = internal_module.compute_state(next_obs_pack)
                
                self.algo.collect_sample(state, next_state, *obs_pack, action, reward, terminated, truncated)
                state, obs_pack = next_state, next_obs_pack
                
                episode_tracker.update(
                    reward=reward,
                    done=terminated|truncated
                )
                logger.push_tree({'Environment':env_info})
                logger.push('Policy/std', self.algo.actor_critic.mean_std())
            
            # ppo update
            algo_info = self.algo.update(i_iter / train_iters)
            
            # save checkpoint
            if (self.checkpoint_interval is not None) and i_iter % self.checkpoint_interval == 0 :
                self._save_checkpoint()
            
            # save log
            if i_iter % self.log_interval == 0 :
                step_timer.update(self.n_rollout * self.log_interval)

                logger.push_dict(algo_info)
                logger.push('Episode/episode_length', episode_tracker.mean_episode_length)
                logger.push('Episode/episode_return', episode_tracker.mean_episode_return)
                logger.push('Extra/iteration', str(i_iter))
                logger.push('Extra/fps', step_timer.fps)
                logger.push('Extra/spent_times', time_to_str(step_timer.spent_times()))
                logger.push('Extra/remaining_times', time_to_str(step_timer.remaining_times()))

                # logging
                log_str = ''.join(['\n\n\n', '='*60, '\n', str(logger), '='*60, '\n'])
                print(log_str)
                self._save_log(log_str)

                # logging to tensorboard
                for key, value in logger.flatten().items() :
                    if not isinstance(value, float) :
                        continue
                    self.writer.add_scalar(key, value, self.acc_iters)

                episode_tracker.clear()
                logger.clear_data()
        
        # save last model
        self._save_checkpoint()
    
    
    def close(self) :
        self.writer.close()
