"""
Utilities for evaluating the global reward function.

Reward prioritizes information gain (reduction in entropy) while
penalizing elapsed time via the kappa constant and costly communication.
"""

import numpy as np


def _as_probability_vector(belief):
    """
    Convert an input into a 1-D probability vector and validate basic properties.

    Raises:
        ValueError: if the belief is empty, negative, or does not sum to 1
                    within a small tolerance.
    """
    probabilities = np.asarray(belief, dtype=float).ravel()

    if probabilities.size == 0:
        raise ValueError("Belief vectors must contain at least one element.")

    if np.any(probabilities < 0.0):
        raise ValueError("Belief vectors cannot contain negative values.")

    total = probabilities.sum()
    if not np.isclose(total, 1.0, atol=1e-8):
        raise ValueError(
            "Belief vectors must sum to 1 (received sum={:.6f}).".format(total)
        )

    return probabilities


def compute_entropy(belief, eps=1e-12):
    """
    Shannon entropy for a discrete belief distribution.

    Args:
        belief: 1-D iterable representing the probability of each grid cell.
        eps: Lower bound to avoid log(0); must be positive.

    Returns:
        The entropy H(belief).
    """
    if eps <= 0:
        raise ValueError("eps must be strictly positive.")

    probabilities = _as_probability_vector(belief)
    support = probabilities > 0.0

    if not np.any(support):
        return 0.0

    clipped = np.clip(probabilities[support], eps, 1.0)
    entropy = -np.sum(probabilities[support] * np.log(clipped))
    return float(entropy)


def information_gain(
    prev_belief,
    next_belief,
    eps=1e-12,
):
    """
    Information gain between successive beliefs, Delta H = H(b_t) - H(b_{t+1}).
    """
    return float(
        compute_entropy(prev_belief, eps=eps)
        - compute_entropy(next_belief, eps=eps)
    )


def global_reward(
    prev_belief,
    next_belief,
    kappa,
    comm_cost,
    communicated,
    eps=1e-12,
):
    """
    Global reward encouraging uncertainty reduction while penalizing costs.

    Args:
        prev_belief: Belief vector before executing the joint action.
        next_belief: Belief vector after executing the joint action.
        kappa: Positive cost proportional to time elapsed.
        comm_cost: Positive cost applied when communication occurs.
        communicated: Boolean value indicating if agents communicated (I_t^comm).
        eps: Stability constant forwarded to entropy computations.

    Returns:
        Reward value R(s_t, a_t) = Delta H_t - kappa - comm_cost * I_t^{comm}.
    """
    if kappa < 0:
        raise ValueError("kappa must be non-negative.")
    if comm_cost < 0:
        raise ValueError("comm_cost must be non-negative.")

    comm_indicator = int(bool(communicated))
    delta_h = information_gain(prev_belief, next_belief, eps=eps)
    reward = delta_h - float(kappa) - float(comm_cost) * comm_indicator
    return float(reward)


def _demo():
    """Run a small example to illustrate how the reward is computed."""
    prior = np.array([0.6, 0.3, 0.1])
    posterior = np.array([0.8, 0.15, 0.05])

    print("Entropy prior: {}".format(compute_entropy(prior)))
    print("Entropy posterior: {}".format(compute_entropy(posterior)))

    delta_h = information_gain(prior, posterior)
    reward = global_reward(
        prior,
        posterior,
        kappa=0.05,
        comm_cost=0.02,
        communicated=True,
    )

    print("Information gain: {:.4f} nats".format(delta_h))
    print("Reward (with communication): {:.4f}".format(reward))


if __name__ == "__main__":
    _demo()