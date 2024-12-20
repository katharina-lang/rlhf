import torch
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from rlhf.utils.env import make_env


class PPOSetup:
    @staticmethod
    def set_up_writer(run_name, args):
        """
        Richtet TensorBoard Writer für Logging ein.
        """
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        return writer

    @staticmethod
    def set_up_device(args):
        """
        Wählt das Gerät (CPU oder CUDA) für die Berechnungen aus.
        """
        device = torch.device(
            "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
        )
        return device

    @staticmethod
    def set_up_envs(args, run_name):
        """
        Richtet die Gym-Umgebung für den PPO-Agenten ein.
        """
        envs = gym.vector.SyncVectorEnv(
            [
                make_env(args.env_id, i, args.capture_video, run_name, args.gamma)
                for i in range(args.num_envs)
            ]
        )
        assert isinstance(
            envs.single_action_space, gym.spaces.Box
        ), "only continuous action space is supported"
        return envs

    @staticmethod
    def set_up_storage(args, envs, device):
        """
        Erstellt Speicher für Rollouts (Beobachtungen, Aktionen, Belohnungen usw.).
        """
        obs = torch.zeros(
            (args.num_steps, args.num_envs) + envs.single_observation_space.shape
        ).to(device)
        actions = torch.zeros(
            (args.num_steps, args.num_envs) + envs.single_action_space.shape
        ).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(device)
        return obs, actions, logprobs, rewards, dones, values
