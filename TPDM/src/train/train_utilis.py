import os

import torch


def get_kl_beta(beta1, alpha1, beta2, alpha2):
    """
    calculate the KL divergence between two beta distributions
    """
    B1 = torch.special.gammaln(alpha1) + torch.special.gammaln(beta1) - torch.special.gammaln(alpha1 + beta1)
    B2 = torch.special.gammaln(alpha2) + torch.special.gammaln(beta2) - torch.special.gammaln(alpha2 + beta2)

    kl_div = (
        (B2 - B1)
        + (alpha1 - alpha2) * torch.special.digamma(alpha1)
        + (beta1 - beta2) * torch.special.digamma(beta1)
        - (alpha1 - alpha2 + beta1 - beta2) * torch.special.digamma(alpha1 + beta1)
    )

    return kl_div

def setup_debug():
    import debugpy

    rank = int(os.environ.get("RANK", 0))

    if rank == 0:
        print(f"Debugger listening on rank {rank}")
        debugpy.listen(("0.0.0.0", 5678))
        debugpy.wait_for_client()
        print("Debugger attached")
    else:
        print(f"Running on rank {rank} without debugger")


if __name__ == "__main__":
    alpha1, beta1 = torch.tensor(2.0), torch.tensor(5.0)
    alpha2, beta2 = torch.tensor(3.0), torch.tensor(4.0)

    kl_div = get_kl_beta(alpha1, beta1, alpha2, beta2)
    print("KL divergence:", kl_div.item())
    kl_div = torch.distributions.kl_divergence(
        torch.distributions.Beta(alpha1, beta1), torch.distributions.Beta(alpha2, beta2)
    )
    print("KL divergence:", kl_div.item())
