import math
import torch
from matplotlib import pyplot as plt

EPSILON = 1e-3
CONCENTRATION = 20
ex = math.exp(1)

def get_ref_beta(sigmas_1, num_steps=28):
    t_1 = sigmas_1 / (ex + (1 - ex) * sigmas_1)
    t_2 = torch.clamp(t_1 - 1.0 / num_steps, EPSILON)
    sigmas_2 = ex / (ex + 1 / t_2 - 1)
    mode = sigmas_2 / sigmas_1

    concentration = CONCENTRATION
    alpha = mode * (concentration - 2) + 1
    beta = (1 - mode) * (concentration - 2) + 1

    return alpha, beta

# draw a  beta distribution
def draw_beta(alpha, beta, num_steps=28, save_path=None):
    a = torch.arange(num_steps) / num_steps
    b = torch.distributions.Beta(alpha, beta).log_prob(a).exp()
    maxx = (alpha - 1) / (alpha + beta - 2)
    if save_path is not None:
        plt.figure()
        plt.plot(a, b)
        plt.plot(maxx, 00.0, 'ro')
        plt.savefig(save_path)

if __name__ == "__main__":
    a = torch.arange(1, 29) / 28.0
    sigmas_1 = ex / ( ex + 1 / a - 1)
    alpha, beta = get_ref_beta(sigmas_1)
    for i in range(1, 28):
        draw_beta(alpha[i].item(), beta[i].item(), save_path=f"beta_{i}.png")