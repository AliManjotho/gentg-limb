"""DDPM/DDIM diffusion utilities for Generative Pose Corrector (GPC)."""
from __future__ import annotations
from dataclasses import dataclass
import math, torch

@dataclass
class NoiseSchedule:
    steps: int = 1000
    schedule: str = "cosine"  # "linear" | "cosine" | "sqrt"

    def betas(self, device=None) -> torch.Tensor:
        t = torch.linspace(0, 1, self.steps+1, device=device)
        if self.schedule == "linear":
            b = torch.linspace(1e-4, 0.02, self.steps, device=device)
        elif self.schedule == "sqrt":
            b = torch.linspace(1e-4, 0.02, self.steps, device=device) ** 0.5
        else:  # cosine (improved DDPM)
            s=0.008
            alphas = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
            alphas = alphas / alphas[0]
            b = 1 - (alphas[1:] / alphas[:-1])
        return b.clamp(1e-8, 0.999)

class DiffusionProcess:
    def __init__(self, steps: int = 1000, schedule: str = "cosine"):
        self.steps = steps
        self.betas = NoiseSchedule(steps, schedule).betas()
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
        """
        if noise is None:
            noise = torch.randn_like(x0)
        a_bar = self.alpha_bars[t].view(-1, *([1]*(x0.dim()-1)))
        return (a_bar.sqrt() * x0 + (1 - a_bar).sqrt() * noise, noise)

    @torch.no_grad()
    def ddim_step(self, x_t: torch.Tensor, t: int, t_prev: int, eps_hat: torch.Tensor, eta: float = 0.0) -> torch.Tensor:
        """
        Deterministic DDIM update (optionally with small stochasticity via eta).
        """
        a_t = self.alpha_bars[t]
        a_prev = self.alpha_bars[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=x_t.device)
        x0_hat = (x_t - (1 - a_t).sqrt() * eps_hat) / a_t.sqrt()
        dir_term = (1 - a_prev).sqrt() * eps_hat
        if eta > 0:
            z = torch.randn_like(x_t)
            sigma = eta * ((1 - a_prev) / (1 - a_t)).sqrt() * (1 - a_t / a_prev).sqrt()
            return a_prev.sqrt() * x0_hat + dir_term + sigma * z
        return a_prev.sqrt() * x0_hat + dir_term
