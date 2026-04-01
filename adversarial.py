# ============================================================
# SHIELD FL — adversarial.py
# Poisoning attack and differential privacy defence
# Import with: from adversarial import poisoning_attack, differential_privacy
# ============================================================

import numpy as np


def poisoning_attack(weights, biases, strength=2.0, seed=42):
    """
    Simulates a model poisoning attack.

    A malicious client corrupts its weight updates before
    sending them to the server. When aggregated with clean
    clients the global model degrades — it starts
    misclassifying attacks as normal traffic.

    Args:
        weights:  Dense layer weights
        biases:   Dense layer biases
        strength: Noise multiplier (higher = stronger attack)
        seed:     Random seed
    Returns:
        Corrupted weights and biases
    """
    rng = np.random.RandomState(seed)
    p_weights = [w + rng.normal(0, strength * np.std(w), w.shape)
                 for w in weights]
    p_biases  = [b + rng.normal(0, strength * np.std(b), b.shape)
                 for b in biases]
    return p_weights, p_biases


def differential_privacy(weights, biases, noise_scale=0.01, seed=42):
    """
    Applies differential privacy to model weight updates.

    Clips weight norms and adds calibrated Gaussian noise
    before uploading to the server. This limits how much
    any single client — including a malicious one — can
    shift the global model.

    Reference: Brauneck et al. (2023) — cited in SHIELD paper.

    Args:
        weights:     Weight updates to protect
        biases:      Bias updates to protect
        noise_scale: Gaussian noise std
        seed:        Random seed
    Returns:
        Privacy-protected weights and biases
    """
    rng = np.random.RandomState(seed)
    dp_weights = [w / max(np.linalg.norm(w), 1.0) +
                  rng.normal(0, noise_scale, w.shape)
                  for w in weights]
    dp_biases  = [b / max(np.linalg.norm(b), 1.0) +
                  rng.normal(0, noise_scale, b.shape)
                  for b in biases]
    return dp_weights, dp_biases


if __name__ == '__main__':
    print('Adversarial functions loaded.')
    print('  poisoning_attack()      — simulates malicious client')
    print('  differential_privacy()  — defends against poisoning')
