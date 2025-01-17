from typing import Tuple
import torch
from sampling.epsilon_net import ddim_step, EpsilonNet


def dps(
    initial_noise: torch.Tensor,
    inverse_problem: Tuple,
    epsilon_net: EpsilonNet,
    gamma: float = 1.0,
    eta: float = 1.0,
):
    """Use DPS algorithm to solve an inverse problem.

    This is an implementation of [1].
    Use ``gamma`` to control the strength of the gradient perturbation.

    Note
    ----
    - Use ``initial_noise`` to set the number of samples ``(n_samples, *shape_of_data)``.

    References
    ----------
    .. [1] Chung, Hyungjin, et al. "Diffusion posterior sampling for general noisy inverse problems."
    arXiv preprint arXiv:2209.14687 (2022).

    """
    obs, H_func, std = inverse_problem
    A = H_func.H
    shape = (initial_noise.shape[0], *(1,) * len(initial_noise.shape[1:]))

    def pot_func(x):
        return -torch.norm(obs.reshape(1, -1) - A(x)) ** 2.0

    def error(x):
        return torch.norm(obs.reshape(1, -1) - A(x), dim=-1)
    
    def proj(x, y, degradation_operator):
        s = degradation_operator.singulars().unsqueeze(0)
        U, Ut, V, Vt, H = degradation_operator.U, degradation_operator.Ut, degradation_operator.V, degradation_operator.Vt, degradation_operator.H
        y = y.unsqueeze(0)
        v_input = torch.multiply(s**(-1), Ut(H(x) - y))
        v_input = v_input.repeat(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3] // (x.shape[1] * x.shape[2]))#[:, :3 * 256 * 256]#.view(1, 3 * 256 * 256)
        # V needs input (1,3*256*256)
        return x - V(v_input).reshape(x.shape)
    k = len(epsilon_net.timesteps)
    sample = initial_noise
    for i in range(len(epsilon_net.timesteps) - 1, 1, -1):
        t, t_prev = epsilon_net.timesteps[i], epsilon_net.timesteps[i - 1]

        sample.requires_grad_()
        e_t = epsilon_net.predict_x0(sample, t)
        grad_norm = error(e_t).reshape(*shape)
        pot_val = pot_func(e_t)
        grad_pot = torch.autograd.grad(pot_val, sample)[0]

        sample = ddim_step(
            x=sample, epsilon_net=epsilon_net, t=t, t_prev=t_prev, eta=eta, e_t=e_t
        ).detach()
        proj_sample = proj(sample, obs, H_func)
        sample = int(1-(i/k)) * sample + int(i/k) * proj_sample

        # gradient step
        grad_pot = gamma * grad_pot / grad_norm
        sample = sample + grad_pot

    # last diffusion step
    sample.requires_grad_()
    grad_lklhd = torch.autograd.grad(pot_func(sample), sample)[0]
    grad_norm = error(sample).reshape(*shape)
    grad_lklhd = (gamma / grad_norm) * grad_lklhd

    sample = epsilon_net.predict_x0(sample, epsilon_net.timesteps[1]) + grad_lklhd

    return sample.detach()
