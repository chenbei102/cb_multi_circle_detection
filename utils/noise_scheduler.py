# -*- coding: utf-8 -*-

"""
This module defines a class for a linear noise scheduler used in diffusion models.  
"""

import numpy as np


class LinearNoiseScheduler:
    """
    A linear noise scheduler from diffusion models.

    This scheduler defines a linear noise schedule, computing beta 
    (variance) values and their corresponding alpha values across timesteps. 
    These values are used for adding noise in diffusion processes.

    Attributes:
        num_timesteps (int): Number of timesteps in the diffusion process.
        beta_start (float): Initial beta value (variance schedule).
        beta_end (float): Final beta value (variance schedule).
        betas (numpy.ndarray): Linearly spaced beta values across timesteps.
        alphas (numpy.ndarray): Alpha values computed as (1 - betas).
        alpha_cum_prod (numpy.ndarray): Cumulative product of alpha values.
        sqrt_alpha_cum_prod (numpy.ndarray): Square root of cumulative alpha product.
        sqrt_one_minus_alpha_cum_prod (numpy.ndarray): Square root of (1 - cumulative alpha product).
    """

    def __init__(self, num_timesteps, beta_start, beta_end):
        """
        Initialize the LinearNoiseScheduler with the given parameters.

        Args:
            num_timesteps (int): The number of timesteps in the diffusion process.
            beta_start (float): The starting value of beta.
            beta_end (float): The ending value of beta.
        """

        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = np.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cum_prod = np.cumprod(self.alphas)
        self.sqrt_alpha_cum_prod = np.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = np.sqrt(1.0 - self.alpha_cum_prod)

    def add_noise(self, original, t, noise=None):
        """
        Add noise to the input data based on the specified timestep.

        Args:
            original (numpy.array): The original input data.
            t (int): The timestep index, which determines the noise level.
            noise (numpy.array, optional): The noise to be added. If None,
                noise is generated internally.

        Returns:
            numpy.array: The resulting data with added noise.
        """

        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod[t]
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod[t]

        if noise is None:
            noise = np.random.normal(size=original.shape)

        return sqrt_alpha_cum_prod * original + sqrt_one_minus_alpha_cum_prod * noise
