from argparse import ArgumentParser
parser = ArgumentParser(description='')
parser.add_argument('env_name', type=str)
parser.add_argument('model_name', type=str)
parser.add_argument('run_mode', choices=['train', 'play', 'test'])
parser.add_argument('--asym', action='store_true')
parser.add_argument('--stochastic', action='store_true')
parser.add_argument('--terrain-level', action='store_true')
parser.add_argument('--seed', type=int, default=None)

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

ENV_NAME = args_cli.env_name
RUN_PATH = 'runs/%s'%args_cli.model_name
MODEL_PATH = 'models/%s.pt'%args_cli.model_name
RUN_MODE = args_cli.run_mode
IS_SYM = not args_cli.asym
STOCHASTIC = args_cli.stochastic
TERRAIN_LEVEL = args_cli.terrain_level
SEED = args_cli.seed

from my_simple_rl.runner import PPORunner
from my_simple_rl.ppo import PPO, PPOCfg
from my_simple_rl.modules import MlpActorCritic
from my_simple_rl.internal_module import InternalModule, MLPEstimator
from my_simple_rl.env import LeggedEnvWrapper

import sys
sys.path.append('D:\ChanJun\workspace\IsaacLab-Projects\Legged-Lab')
import legged_lab.task
import torch as th


def set_seed(seed:int) :
    import os, random, numpy, torch

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() :
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    torch.use_deterministic_algorithms(True)


def main() :
    if SEED is not None :
        set_seed(SEED)
    env = legged_lab.task.make(ENV_NAME)
    env = LeggedEnvWrapper(env, reward_scale=1/50)

    n_policy_obs = env.n_obs * env.n_history + env.n_privileged_obs
    n_estimator_obs = env.n_obs * env.n_history
    
    actor_critic = MlpActorCritic(
        n_obs=n_policy_obs,
        n_action=env.n_action,
        init_std=1.0,
        net_arch=[512, 256, 128],
        activ_fn=th.nn.ELU,
        is_sym=IS_SYM,
    )
    ppo_cfg = PPOCfg(
        n_rollout=50,
        n_epoch=5,
        n_minibatch=4,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=1e-3,
        desired_kl=0.01,
        ratio_clip_param=0.2,
        value_clip_param=0.2,
        grad_norm_clip=1.0,
        normalize_advantage=True,
        entropy_loss_coeff=0.01,
        value_loss_coeff=1.0,
        internal_n_minibatch=16,
    )
    internal_module = InternalModule(
        estimator=MLPEstimator(
            input_dim=n_estimator_obs,
            output_dim=env.n_privileged_obs,
            net_arch=[512, 256, 128],
            activ_fn=th.nn.ReLU,
        ),
        learning_rate=0.001,
        device=env.device,
    )
    ppo = PPO(env.spec, actor_critic, internal_module, ppo_cfg)
    
    # train
    if RUN_MODE == 'train' :
        runner = PPORunner(
            env=env,
            algo=ppo,
            run_dir=RUN_PATH,
            log_interval=1,
            checkpoint_interval=300,
        )
        runner.train(3000)
        ppo.save(MODEL_PATH)
    
    # test/play
    elif RUN_MODE == 'test' or RUN_MODE == 'play' or RUN_MODE == 'control' :
        if RUN_MODE == 'play' or RUN_MODE == 'control' :
            ppo.load(MODEL_PATH)
        
        obs_pack, info = env.reset()
        state = internal_module.compute_state(obs_pack)
        step_cnt = 0
        
        while simulation_app.is_running() :
            next_obs_pack, rwd, ter, tru, info = env.step(
                ppo.act(state, deterministic=(not STOCHASTIC))
            )
            next_state = internal_module.compute_state(next_obs_pack)
            state = next_state

            if TERRAIN_LEVEL and (step_cnt % 10 == 0) :
                x_grid_num = env.env.command_manager.cfg.x_grid_num
                y_grid_num = env.env.command_manager.cfg.y_grid_num
                n_grid = env.env.command_manager.n_grid
                grid_ids = env.env.command_manager._grid_ids
                terrain_levels = env.env.scene.terrain.terrain_levels

                grid_level_sum = th.bincount(grid_ids, terrain_levels, n_grid)
                grid_level_cnt = th.bincount(grid_ids, minlength=n_grid).clip(min=1)
                grid_level = grid_level_sum / grid_level_cnt
                grid_level = grid_level.view(x_grid_num, y_grid_num)
                
                output = []
                output.append('\n\n\n\n')
                output.append('global terrain level : %5.3f\n'%terrain_levels.to(th.float32).mean().item())
                output.append('grid terrain level :\n')
                for rows in reversed(grid_level.tolist()) :
                    output.extend(['%4.2f  '% ele for ele in rows])
                    output.append('\n')
                output = ''.join(output)
                print(output)

            step_cnt += 1
    
    env.close()


if __name__ == '__main__' :
    main()
    simulation_app.close()
