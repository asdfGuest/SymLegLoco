import torch as th
from typing import Dict

from my_simple_rl.env import EnvSpec
from my_simple_rl.ppo.buffer import PPOBuffer
from my_simple_rl.ppo.config import PPOCfg
from my_simple_rl.internal_module import InternalModule
from simple_rl.modules.modules import BaseActorCritic


class PPO:
    def __init__(
            self,
            env_spec:EnvSpec,
            actor_critic:BaseActorCritic,
            internal_module:InternalModule,
            cfg:PPOCfg,
        ) :
        self.env_spec = env_spec
        self.actor_critic = actor_critic
        self.internal_module = internal_module
        self.cfg = cfg

        self.learning_rate = self.cfg.learning_rate
        self.buffer = PPOBuffer(self.cfg, keys=[
            'state',
            'obs',
            'privileged_obs',
            'action',
            'action_mean',
            'action_std',
            'action_logprob',
            # essential keys
            'reward',
            'done',
            'value',
            'advantage',
            'returns'
        ])
        
        self.actor_critic.to(self.env_spec.device)
        self.optimizer = th.optim.Adam(
            params=self.actor_critic.parameters(),
            lr=self.learning_rate
        )
    
    def save(self, path:str) :
        th.save({
            'learning_rate' : self.learning_rate,
            'actor_critic_state_dict' : self.actor_critic.state_dict(),
            'internal_module_state_dict' : self.internal_module.state_dict(),
            'optimizer_state_dict' : self.optimizer.state_dict(),
        }, path)

    def load(self, path:str) :
        state_dict = th.load(path, map_location=self.env_spec.device)

        self.learning_rate = state_dict['learning_rate']
        self.actor_critic.load_state_dict(state_dict['actor_critic_state_dict'])
        self.internal_module.load_state_dict(state_dict['internal_module_state_dict'])
        
        self.optimizer = th.optim.Adam(
            params=self.actor_critic.parameters(),
            lr=self.learning_rate
        )
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    def act(self, state:th.Tensor, deterministic=False) :
        with th.no_grad():
            self._last_act_pdf = self.actor_critic.policy.compute(state)
            return self._last_act_pdf.mean if deterministic else self._last_act_pdf.sample()

    def collect_sample(
            self,
            state:th.Tensor,
            next_state:th.Tensor,
            obs:th.Tensor,
            privileged_obs:th.Tensor,
            action:th.Tensor,
            reward:th.Tensor,
            terminated:th.Tensor,
            truncated:th.Tensor,
        ) :
        with th.no_grad():
            pdf = self._last_act_pdf
            action_logprob = pdf.log_prob(action)
            value = self.actor_critic.value.compute(state)
            reward = reward + truncated * self.cfg.gamma * value

        self.buffer.push(
            state=state,
            obs=obs,
            privileged_obs=privileged_obs,
            action=action,
            action_mean=pdf.mean,
            action_std=pdf.std,
            action_logprob=action_logprob,
            reward=reward,
            done=terminated|truncated,
            value=value,
            advantage=None,
            returns=None,
        )
        self.next_state = next_state.detach()

    def _update_learning_rate(self, policy_kld:float) :
        desired_kl = self.cfg.desired_kl

        if desired_kl is not None :

            if policy_kld > desired_kl * 2.0 :
                self.learning_rate = max(self.learning_rate / 1.5, 1e-5)
            elif policy_kld < desired_kl / 2.0 :
                self.learning_rate = min(self.learning_rate * 1.5, 1e-2)
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate

    def update(self, train_rate:float) -> Dict[str,float]:

        _wrapper = lambda x: x if isinstance(x, float) or x is None else x(train_rate)
        ratio_clip_param = _wrapper(self.cfg.ratio_clip_param)
        value_clip_param = _wrapper(self.cfg.value_clip_param)
        entropy_loss_coeff = _wrapper(self.cfg.entropy_loss_coeff)
        
        info:Dict[str, list] = {}
        def add_info(key:str, value) :
            if key not in info :
                info[key] = []
            info[key].append(value)
        
        with th.no_grad() :
            next_value = self.actor_critic.value.compute(self.next_state)
        self.buffer.compute_gae(next_value)
        
        for minibatchs in self.buffer.get_minibatch_generator() :
            # get minibatch
            state, obs, privileged_obs, action_old, *pdf_old, action_old_logprob, reward, done, value_old, advantage, returns = minibatchs
            pdf_old = self.actor_critic.policy.get_pdf_cls()(*pdf_old)

            if self.cfg.normalize_advantage :
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            
            pdf = self.actor_critic.policy.compute(state)
            # adaptive learning rate
            if self.cfg.desired_kl is not None :
                with th.no_grad() :
                    policy_kld = self.actor_critic.policy.get_pdf_cls().kld(pdf_old, pdf).mean().item()
                self._update_learning_rate(policy_kld)
            # policy loss
            action_logprob = pdf.log_prob(action_old)
            ratio = th.exp(action_logprob - action_old_logprob)
            ratio_clip = th.clip(ratio, 1.-ratio_clip_param, 1.+ratio_clip_param)
            policy_loss = th.mean(-th.min(ratio * advantage, ratio_clip * advantage))
            # value loss
            value = self.actor_critic.value.compute(state)
            if value_clip_param is not None :
                value_clip = value_old + th.clip(value-value_old, -value_clip_param, value_clip_param)
                value_loss = th.mean(th.max(
                    (returns - value)**2,
                    (returns - value_clip)**2
                ))
            else :
                value_loss = th.mean((value - returns)**2)
            # entropy bonus
            entropy_bonus = th.mean(pdf.entropy())
            # final loss
            loss = policy_loss + value_loss * self.cfg.value_loss_coeff - entropy_bonus * entropy_loss_coeff
            # optimization
            self.optimizer.zero_grad()
            loss.backward()
            if self.cfg.grad_norm_clip is not None :
                th.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.cfg.grad_norm_clip)
            self.optimizer.step()
            # logging
            if self.cfg.desired_kl is not None :
                add_info('PPO/learning_rate', self.learning_rate)
            add_info('PPO/policy_loss', policy_loss.item())
            add_info('PPO/value_loss', value_loss.item())
            add_info('PPO/entropy_bonus', entropy_bonus.item())

            # update internal module
            iternal_minibatch_size = obs.shape[0] // self.cfg.internal_n_minibatch
            batch_tensors = [obs, privileged_obs]

            for idx in range(self.cfg.internal_n_minibatch) :
                low = iternal_minibatch_size * idx
                high = iternal_minibatch_size * (idx + 1)
                minibatch_tensors = [tensor[low:high] for tensor in batch_tensors]
                
                tmp = self.internal_module.update(minibatch_tensors)
                for key, value in tmp.items() :
                    add_info(key, value)
        
        self.buffer.clear()

        _mean = lambda x: sum(x) / len(x)
        for key in info :
            info[key] = _mean(info[key])
        return info
