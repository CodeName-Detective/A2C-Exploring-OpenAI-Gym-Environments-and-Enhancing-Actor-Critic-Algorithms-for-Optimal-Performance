import pickle
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from tqdm.notebook import tqdm_notebook as tqdm

from src.utils import DEVICE


class A2C_Categorical_Action_V1:
    def __init__(self, env, env_name, Actor_Function, Critic_Function, lr_actor, lr_q_value_critic, lr_value_critic, discount_factor, convergence_value, convergence_len=20):
        self.env = env
        self.observation_space = self.env.observation_space.shape[0] if self.env.observation_space.shape else self.env.observation_space.n
        self.action_space = self.env.action_space.n
        
        self.env_name = env_name
        
        self.discount_factor = discount_factor
        
        self.convergence_value = convergence_value
        self.convergence_len = convergence_len
        self.convergence_tracker = deque(maxlen=self.convergence_len)
        
        self.actor = Actor_Function(self.observation_space, self.action_space).to(DEVICE)
        self.q_critic = Critic_Function(self.observation_space, self.action_space).to(DEVICE)
        self.value_critic = Critic_Function(self.observation_space, 1).to(DEVICE)
        
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr = lr_actor)
        self.q_critic_optim = torch.optim.Adam(self.q_critic.parameters(), lr = lr_q_value_critic)
        self.value_critic_optim = torch.optim.Adam(self.value_critic.parameters(), lr = lr_value_critic)
        
        
        self.reward_across_episdoes = []
    
    def _save_model(self):
        with open(f'sadapala_{self.assignment_part}_{self.env_name}.pickle', 'wb') as f:
            pickle.dump(self, f)
    
    def choose_action(self, state):
        action_dist = self.actor(state)
        action_dist = torch.distributions.categorical.Categorical(probs=action_dist)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action)
    
    def evaluate(self, episodes):
        prog_bar = tqdm(range(episodes), desc='Evaluation:', disable=False)
        rewards_across_episodes = []
        for _ in prog_bar:
            observation, _ = self.env.reset()
            reward_per_episode = 0
            terminated = False
            observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            while not terminated:
                current_action,  _ = self.choose_action(observation)
                new_observation, reward, terminated, truncated, _ = self.env.step(current_action)
                terminated = terminated or truncated
                new_observation = torch.tensor(new_observation, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                reward_per_episode += reward
                observation = new_observation
            
            rewards_across_episodes.append(reward_per_episode)
        
        return rewards_across_episodes
    
    def train(self, episodes, writer):
        prog_bar = tqdm(range(episodes), desc='Training Episode:', disable=False)
        step = 0
        for episode in prog_bar:
            observation, _ = self.env.reset()
            reward_per_episode = 0
            actor_loss_per_episode = 0
            q_critic_error_per_episode = 0
            state_critic_error_per_episode = 0
            counter = 0
            terminated = False
            observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            while not terminated:
                current_action,  current_action_log_prob = self.choose_action(observation)
                new_observation, reward, terminated, truncated, _ = self.env.step(current_action)
                
                terminated = terminated or truncated
                new_observation = torch.tensor(new_observation, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                reward_per_episode += reward
                
                new_action,  _ = self.choose_action(new_observation)
                
                # Train
                state_value = self.value_critic(observation)
                q_value = self.q_critic(observation)[0,current_action]
                
                # Update Actor
                advantage_value_action = q_value - state_value
                actor_loss = - current_action_log_prob * advantage_value_action.detach()
                actor_loss_per_episode += actor_loss.item()
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()
                
                # Update q_critic
                action_value_next = self.q_critic(new_observation)[0,new_action]
                #action_value_next = torch.max(self.q_critic(new_observation))
                q_target = reward + (self.discount_factor*(action_value_next)*(1-int(terminated)))
                q_critic_error = torch.nn.functional.mse_loss(q_value, q_target)
                q_critic_error_per_episode += q_critic_error.item()
                self.q_critic_optim.zero_grad()
                q_critic_error.backward()
                self.q_critic_optim.step()
                
                # Update value_critic
                state_value_next = self.value_critic(new_observation)
                state_value_target = reward + (self.discount_factor*(state_value_next)*(1-int(terminated)))
                state_critic_error = torch.nn.functional.mse_loss(state_value, state_value_target)
                state_critic_error_per_episode += state_critic_error.item()
                self.value_critic_optim.zero_grad()
                state_critic_error.backward()
                self.value_critic_optim.step()
                
                observation = new_observation
                
                step += 1
                counter += 1
            
            writer.add_scalar("reward/reward_per_episode", reward_per_episode, episode)
            writer.add_scalar("loss/actor_loss_per_episode", actor_loss_per_episode/counter, episode)
            writer.add_scalar("loss/q_critic_error_per_episode", q_critic_error_per_episode/counter, episode)
            writer.add_scalar("loss/state_critic_error_per_episode", state_critic_error_per_episode/counter, episode)
            
            self.convergence_tracker.append(reward_per_episode)
            self.reward_across_episdoes.append(reward_per_episode)
            
            prog_bar.set_postfix_str(
                    f"Convergence Criteria-Mean:{np.mean(self.convergence_tracker)}, Min:{np.min(self.convergence_tracker)}, Max:{np.max(self.convergence_tracker)}"
                    )
            
            if np.mean(self.convergence_tracker) >= self.convergence_value:
                self._save_model()
                break
            
            self._save_model()
        
        writer.flush()
        writer.close()



class A2C_Categorical_Action_V2(A2C_Categorical_Action_V1):
    
    def train(self, episodes, writer):
        prog_bar = tqdm(range(episodes), desc='Training Episode:', disable=False)
        step = 0
        for episode in prog_bar:
            observation, _ = self.env.reset()
            reward_per_episode = 0
            actor_loss_per_episode = 0
            q_critic_error_per_episode = 0
            state_critic_error_per_episode = 0
            counter = 0
            terminated = False
            observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            while not terminated:
                current_action,  current_action_log_prob = self.choose_action(observation)
                new_observation, reward, terminated, truncated, _ = self.env.step(current_action)
                
                terminated = terminated or truncated
                new_observation = torch.tensor(new_observation, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                reward_per_episode += reward
                
                
                # Train
                state_value = self.value_critic(observation)
                q_value = self.q_critic(observation)[0,current_action]
                
                # Update Actor
                advantage_value_action = q_value - state_value
                actor_loss = - current_action_log_prob * advantage_value_action.detach()
                actor_loss_per_episode += actor_loss.item()
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()
                
                # Update q_critic
                action_value_next = torch.max(self.q_critic(new_observation))
                q_target = reward + (self.discount_factor*(action_value_next)*(1-int(terminated)))
                q_critic_error = torch.nn.functional.mse_loss(q_value, q_target)
                q_critic_error_per_episode += q_critic_error.item()
                self.q_critic_optim.zero_grad()
                q_critic_error.backward()
                self.q_critic_optim.step()
                
                # Update value_critic
                state_value_next = self.value_critic(new_observation)
                state_value_target = reward + (self.discount_factor*(state_value_next)*(1-int(terminated)))
                state_critic_error = torch.nn.functional.mse_loss(state_value, state_value_target)
                state_critic_error_per_episode += state_critic_error.item()
                self.value_critic_optim.zero_grad()
                state_critic_error.backward()
                self.value_critic_optim.step()
                
                observation = new_observation
                
                step += 1
                counter += 1
            
            writer.add_scalar("reward/reward_per_episode", reward_per_episode, episode)
            writer.add_scalar("loss/actor_loss_per_episode", actor_loss_per_episode/counter, episode)
            writer.add_scalar("loss/q_critic_error_per_episode", q_critic_error_per_episode/counter, episode)
            writer.add_scalar("loss/state_critic_error_per_episode", state_critic_error_per_episode/counter, episode)
            
            self.convergence_tracker.append(reward_per_episode)
            self.reward_across_episdoes.append(reward_per_episode)
            
            prog_bar.set_postfix_str(
                    f"Convergence Criteria-Mean:{np.mean(self.convergence_tracker)}, Min:{np.min(self.convergence_tracker)}, Max:{np.max(self.convergence_tracker)}"
                    )
            
            if np.mean(self.convergence_tracker) >= self.convergence_value:
                self._save_model()
                break
            
            self._save_model()
        
        writer.flush()
        writer.close()

class A2C_Categorical_Action_V3:
    def __init__(self, env, env_name, Actor_Function, Critic_Function, lr_actor, lr_value_critic, discount_factor, convergence_value, convergence_len=20):
        self.env = env
        self.observation_space = self.env.observation_space.shape[0] if self.env.observation_space.shape else self.env.observation_space.n
        self.action_space = self.env.action_space.n
        
        self.env_name = env_name
        
        self.discount_factor = discount_factor
        
        self.convergence_value = convergence_value
        self.convergence_len = convergence_len
        self.convergence_tracker = deque(maxlen=self.convergence_len)
        
        self.actor = Actor_Function(self.observation_space, self.action_space).to(DEVICE)
        self.value_critic = Critic_Function(self.observation_space, 1).to(DEVICE)
        
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr = lr_actor)
        self.value_critic_optim = torch.optim.Adam(self.value_critic.parameters(), lr = lr_value_critic)
        
        
        self.reward_across_episdoes = []
    
    def _save_model(self):
        with open(f'sadapala_{self.assignment_part}_{self.env_name}.pickle', 'wb') as f:
            pickle.dump(self, f)
    
    def choose_action(self, state):
        action_dist = self.actor(state)
        action_dist = torch.distributions.categorical.Categorical(probs=action_dist)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action)
    
    def evaluate(self, episodes):
        prog_bar = tqdm(range(episodes), desc='Evaluation:', disable=False)
        rewards_across_episodes = []
        for _ in prog_bar:
            observation, _ = self.env.reset()
            reward_per_episode = 0
            terminated = False
            observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            while not terminated:
                current_action,  _ = self.choose_action(observation)
                new_observation, reward, terminated, truncated, _ = self.env.step(current_action)
                terminated = terminated or truncated
                new_observation = torch.tensor(new_observation, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                reward_per_episode += reward
                observation = new_observation
            
            rewards_across_episodes.append(reward_per_episode)
        
        return rewards_across_episodes
    
    def train(self, episodes, writer):
        prog_bar = tqdm(range(episodes), desc='Training Episode:', disable=False)
        step = 0
        for episode in prog_bar:
            observation, _ = self.env.reset()
            reward_per_episode = 0
            actor_loss_per_episode = 0
            q_critic_error_per_episode = 0
            state_critic_error_per_episode = 0
            counter = 0
            terminated = False
            observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            while not terminated:
                current_action,  current_action_log_prob = self.choose_action(observation)
                new_observation, reward, terminated, truncated, _ = self.env.step(current_action)
                
                terminated = terminated or truncated
                new_observation = torch.tensor(new_observation, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                reward_per_episode += reward
                
                # Train
                state_value = self.value_critic(observation)
                state_value_next = self.value_critic(new_observation)
                state_value_target = reward + (self.discount_factor*(state_value_next)*(1-int(terminated)))
                
                # Update Actor
                advantage_value_action = state_value_target - state_value
                actor_loss = - current_action_log_prob * advantage_value_action.detach()
                actor_loss_per_episode += actor_loss.item()
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()
                
                
                # Update value_critic
                state_critic_error = torch.nn.functional.mse_loss(state_value, state_value_target)
                state_critic_error_per_episode += state_critic_error.item()
                self.value_critic_optim.zero_grad()
                state_critic_error.backward()
                self.value_critic_optim.step()
                
                observation = new_observation
                
                step += 1
                counter += 1
            
            writer.add_scalar("reward/reward_per_episode", reward_per_episode, episode)
            writer.add_scalar("loss/actor_loss_per_episode", actor_loss_per_episode/counter, episode)
            writer.add_scalar("loss/state_critic_error_per_episode", state_critic_error_per_episode/counter, episode)
            
            self.convergence_tracker.append(reward_per_episode)
            self.reward_across_episdoes.append(reward_per_episode)
            
            prog_bar.set_postfix_str(
                    f"Convergence Criteria-Mean:{np.mean(self.convergence_tracker)}, Min:{np.min(self.convergence_tracker)}, Max:{np.max(self.convergence_tracker)}"
                    )
            
            if np.mean(self.convergence_tracker) >= self.convergence_value:
                self._save_model()
                break
            
            self._save_model()
        
        writer.flush()
        writer.close()

class A2C_Continious_Action_V3:
    def __init__(self, env, env_name, Actor_Function, Critic_Function, lr_actor, lr_value_critic, discount_factor, convergence_value, convergence_len=20):
        self.env = env
        self.observation_space = self.env.observation_space.shape[0] if self.env.observation_space.shape else self.env.observation_space.n
        self.action_space = self.env.action_space.shape[0]
        
        self.env_name = env_name
        
        self.discount_factor = discount_factor
        
        self.convergence_value = convergence_value
        self.convergence_len = convergence_len
        self.convergence_tracker = deque(maxlen=self.convergence_len)
        
        self.actor = Actor_Function(self.observation_space, self.action_space).to(DEVICE)
        self.value_critic = Critic_Function(self.observation_space, 1).to(DEVICE)
        
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr = lr_actor)
        self.value_critic_optim = torch.optim.Adam(self.value_critic.parameters(), lr = lr_value_critic)
        
        
        self.reward_across_episdoes = []
    
    def _save_model(self):
        with open(f'sadapala_{self.assignment_part}_{self.env_name}.pickle', 'wb') as f:
            pickle.dump(self, f)
    
    def choose_action(self, state):
        eps = 1e-6
        mu, sigma = self.actor(state)
        action_dist = torch.distributions.normal.Normal(mu, sigma+eps)
        action = action_dist.sample()
        action = torch.clamp(action, self.env.action_space.low.item(), self.env.action_space.high.item())
        return action.squeeze(0).cpu().numpy(), action_dist.log_prob(action)
    
    def evaluate(self, episodes):
        prog_bar = tqdm(range(episodes), desc='Evaluation:', disable=False)
        rewards_across_episodes = []
        for _ in prog_bar:
            observation, _ = self.env.reset()
            reward_per_episode = 0
            terminated = False
            observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            while not terminated:
                current_action,  _ = self.choose_action(observation)
                new_observation, reward, terminated, truncated, _ = self.env.step(current_action)
                terminated = terminated or truncated
                new_observation = torch.tensor(new_observation, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                reward_per_episode += reward
                observation = new_observation
            
            rewards_across_episodes.append(reward_per_episode)
        
        return rewards_across_episodes
    
    def train(self, episodes, writer):
        prog_bar = tqdm(range(episodes), desc='Training Episode:', disable=False)
        step = 0
        for episode in prog_bar:
            observation, _ = self.env.reset()
            reward_per_episode = 0
            actor_loss_per_episode = 0
            q_critic_error_per_episode = 0
            state_critic_error_per_episode = 0
            counter = 0
            terminated = False
            observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            while not terminated:
                current_action,  current_action_log_prob = self.choose_action(observation)
                new_observation, reward, terminated, truncated, _ = self.env.step(current_action)
                
                terminated = terminated or truncated
                new_observation = torch.tensor(new_observation, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                reward_per_episode += reward
                
                # Train
                state_value = self.value_critic(observation)
                state_value_next = self.value_critic(new_observation)
                state_value_target = reward + (self.discount_factor*(state_value_next)*(1-int(terminated)))
                
                # Update Actor
                advantage_value_action = state_value_target - state_value
                actor_loss = - torch.mean(current_action_log_prob * advantage_value_action.detach())
                actor_loss_per_episode += actor_loss.item()
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()
                
                
                # Update value_critic
                state_critic_error = torch.nn.functional.mse_loss(state_value, state_value_target)
                state_critic_error_per_episode += state_critic_error.item()
                self.value_critic_optim.zero_grad()
                state_critic_error.backward()
                self.value_critic_optim.step()
                
                observation = new_observation
                
                step += 1
                counter += 1
            
            writer.add_scalar("reward/reward_per_episode", reward_per_episode, episode)
            writer.add_scalar("loss/actor_loss_per_episode", actor_loss_per_episode/counter, episode)
            writer.add_scalar("loss/state_critic_error_per_episode", state_critic_error_per_episode/counter, episode)
            
            self.convergence_tracker.append(reward_per_episode)
            self.reward_across_episdoes.append(reward_per_episode)
            
            prog_bar.set_postfix_str(
                    f"Convergence Criteria-Mean:{np.mean(self.convergence_tracker)}, Min:{np.min(self.convergence_tracker)}, Max:{np.max(self.convergence_tracker)}"
                    )
            
            if np.min(self.convergence_tracker) >= self.convergence_value:
                self._save_model()
                break
            
            self._save_model()
        
        writer.flush()
        writer.close()

class A2C_Continious_Action_Replay_Buffer_V3:
    def __init__(self, env, env_name, Actor_Function, Critic_Function, lr_actor, lr_value_critic, discount_factor, convergence_value, convergence_len=20):
        self.env = env
        self.observation_space = self.env.observation_space.shape[0] if self.env.observation_space.shape else self.env.observation_space.n
        self.action_space = self.env.action_space.shape[0]
        
        self.env_name = env_name
        
        self.discount_factor = discount_factor
        
        self.convergence_value = convergence_value
        self.convergence_len = convergence_len
        self.convergence_tracker = deque(maxlen=self.convergence_len)
        
        self.actor = Actor_Function(self.observation_space, self.action_space).to(DEVICE)
        self.value_critic = Critic_Function(self.observation_space, 1).to(DEVICE)
        
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr = lr_actor)
        self.value_critic_optim = torch.optim.Adam(self.value_critic.parameters(), lr = lr_value_critic)
        
        
        self.reward_across_episdoes = []
        self.timesteps_across_episodes = []
    
    def _save_model(self):
        with open(f'sadapala_{self.assignment_part}_{self.env_name}.pickle', 'wb') as f:
            pickle.dump(self, f)
    
    def choose_action(self, state):
        eps = 1e-6
        mu, sigma = self.actor(state)
        action_dist = torch.distributions.normal.Normal(mu, sigma+eps)
        action = action_dist.sample()
        action = torch.clamp(action, torch.tensor(self.env.action_space.low).to(DEVICE), torch.tensor(self.env.action_space.high).to(DEVICE))
        return action.squeeze(0).cpu().numpy(), action_dist.log_prob(action)
    
    def evaluate(self, experiments, experiment_length):
        prog_bar = tqdm(range(experiments), desc='Evaluation:', disable=False)
        rewards_across_experiments = []
        for _ in prog_bar:
            time_step = 0
            reward_per_experiment = 0
            observation, _ = self.env.reset()
            observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            while True:
                current_action,  _ = self.choose_action(observation)
                new_observation, reward, terminated, truncated, _ = self.env.step(current_action)
                new_observation = torch.tensor(new_observation, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                terminated = terminated or truncated
                reward_per_experiment += reward
                
                if terminated:
                    observation, _ = self.env.reset()
                    observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                else:
                    observation = new_observation
                
                time_step += 1
                if time_step == experiment_length:
                    rewards_across_experiments.append(reward_per_experiment)
                    break
            
        return rewards_across_experiments
    
    def train(self, episodes, writer):
        prog_bar = tqdm(range(episodes), desc='Training Episode:', disable=False)
        step = 0
        for episode in prog_bar:
            observation, _ = self.env.reset()
            reward_per_episode = 0
            actor_loss_per_episode = 0
            q_critic_error_per_episode = 0
            state_critic_error_per_episode = 0
            counter = 0
            terminated = False
            observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            
            observation_batch = []
            new_observation_batch = []
            reward_batch = []
            terminated_batch = []
            
            current_action_log_prob_batch = []
            
            
            while not terminated:
                current_action,  current_action_log_prob = self.choose_action(observation)
                new_observation, reward, terminated, truncated, _ = self.env.step(current_action)
                
                terminated = terminated or truncated
                new_observation = torch.tensor(new_observation, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                reward_per_episode += reward
                
                observation_batch.append(observation)
                new_observation_batch.append(new_observation)
                reward_batch.append(reward)
                terminated_batch.append(terminated)
                
                current_action_log_prob_batch.append(current_action_log_prob)
                
                
                
                observation = new_observation
                
                step += 1
                counter += 1
                
            
            observation_batch = torch.cat(observation_batch, dim=0)
            new_observation_batch = torch.cat(new_observation_batch, dim=0)
            reward_batch = torch.tensor(reward_batch, dtype=torch.float32).unsqueeze(1).to(DEVICE)
            terminated_batch = torch.tensor(terminated_batch, dtype=torch.float32).unsqueeze(1).to(DEVICE)
            
            current_action_log_prob_batch = torch.cat(current_action_log_prob_batch, dim=0)
           
                
            # Train
            state_value = self.value_critic(observation_batch)
            state_value_next = self.value_critic(new_observation_batch)
            state_value_target = reward_batch + (self.discount_factor*(state_value_next)*(1-terminated_batch))
            
                
                
                
            # Update Actor
            advantage_value_action = state_value_target - state_value
            actor_loss = - torch.mean(current_action_log_prob_batch * advantage_value_action.detach())
            actor_loss_per_episode = actor_loss.item()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            
            
            # Update value_critic
            state_critic_error = torch.nn.functional.mse_loss(state_value, state_value_target)
            state_critic_error_per_episode = state_critic_error.item()
            self.value_critic_optim.zero_grad()
            state_critic_error.backward()
            self.value_critic_optim.step()
                
                
            
            writer.add_scalar("reward/reward_per_episode", reward_per_episode, episode)
            writer.add_scalar("steps/steps_per_episode", counter, episode)
            writer.add_scalar("loss/actor_loss_per_episode", actor_loss_per_episode, episode)
            writer.add_scalar("loss/state_critic_error_per_episode", state_critic_error_per_episode, episode)
            
            self.timesteps_across_episodes.append(counter)
            self.convergence_tracker.append(reward_per_episode)
            self.reward_across_episdoes.append(reward_per_episode)
            
            prog_bar.set_postfix_str(
                    f"Convergence Criteria-Mean:{np.mean(self.convergence_tracker)}, Min:{np.min(self.convergence_tracker)}, Max:{np.max(self.convergence_tracker)}"
                    )
            
            if np.min(self.convergence_tracker) >= self.convergence_value:
                self._save_model()
                break
            
            self._save_model()
        
        writer.flush()
        writer.close()