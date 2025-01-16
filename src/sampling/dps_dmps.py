from typing import Tuple
import torch
from sampling.epsilon_net import ddim_step, EpsilonNet
import numpy as np
from PIL import Image
import os
from PIL import Image, ImageFont, ImageDraw, ImageOps
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from evaluation.perception import LPIPS

def dps_dpms(initial_noise: torch.Tensor,
            inverse_problem: Tuple,
            epsilon_net: EpsilonNet,
            lam: float = 1.0,
            gamma: float = 1.0,
            eta: float = 1.0,
            k: int = 10):
    
    obs, H_func, std = inverse_problem
    A = H_func.H
    shape = (initial_noise.shape[0], *(1,) * len(initial_noise.shape[1:]))

    def pot_func(x):
        return -torch.norm(obs.reshape(1, -1) - A(x)) ** 2.0

    def error(x):
        return torch.norm(obs.reshape(1, -1) - A(x), dim=-1)

    sample = initial_noise
    for i in range(len(epsilon_net.timesteps) - 1, k, -1):
        t, t_prev = epsilon_net.timesteps[i], epsilon_net.timesteps[i - 1]

        sample.requires_grad_()
        e_t = epsilon_net.predict_x0(sample, t)
        grad_norm = error(e_t).reshape(*shape)
        pot_val = pot_func(e_t)
        grad_pot = torch.autograd.grad(pot_val, sample)[0]

        sample = ddim_step(
            x=sample, epsilon_net=epsilon_net, t=t, t_prev=t_prev, eta=eta, e_t=e_t
        ).detach()

        # gradient step
        grad_pot = gamma * grad_pot / grad_norm
        sample = sample + grad_pot

    for i in range(k, 1, -1):
        t, t_prev = epsilon_net.timesteps[i], epsilon_net.timesteps[i - 1]
        sample.requires_grad_()
        e_t = epsilon_net.predict_x0(sample, t)
        grad_value, alpha_t = epsilon_net.approximate_grad_log_likelihood(x_t = sample, t=t_prev, H_funcs=H_func, y=obs, noise_std=std)

        sample = ddim_step(
            x=sample, epsilon_net=epsilon_net, t=t, t_prev=t_prev, eta=eta, e_t=e_t
        ).detach()

        # gradient step
        sample = sample + lam * grad_value * (1-alpha_t)/torch.sqrt(alpha_t)

    # last diffusion step
    sample.requires_grad_()
    grad_value, alpha_t = epsilon_net.approximate_grad_log_likelihood(x_t = sample, t=1, H_funcs=H_func, y=obs, noise_std=std)

    sample = epsilon_net.predict_x0(sample, epsilon_net.timesteps[1]) + lam * grad_value * (1-alpha_t)/torch.sqrt(alpha_t)

    return sample.detach()