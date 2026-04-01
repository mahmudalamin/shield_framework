# ============================================================
# SHIELD FL — aggregation.py
# Three aggregation methods from Velasquez Restrepo & Luo (2025)
# Import with: from aggregation import aggregate
# ============================================================

import numpy as np


def aggregate(all_weights, all_biases, scores,
              mode='fedavg', se_norms=None):
    """
    Aggregates model weights from all clients.
    Formulas taken directly from Velasquez Restrepo & Luo (2025).

    ── FedAvg ────────────────────────────────────────────────
    G_w = (1 / n) × Σ W_i
    Equal weight to every client.

    ── Performance-based ─────────────────────────────────────
    beta  = Σ (1 / score_i)
    phi_i = (1 / score_i) / beta
    G_w   = Σ phi_i × W_i
    Clients with higher AUC get more influence.

    ── DQ-Fed ────────────────────────────────────────────────
    G_w = Σ (NP_i × SE_norm_i × W_i) / Σ NP_i
    Clients with cleaner data and stable training get more influence.

    Args:
        all_weights: List of Dense weights from each client
        all_biases:  List of Dense biases from each client
        scores:      Score per client (AUC or NP depending on mode)
        mode:        'fedavg' | 'performance' | 'dqa'
        se_norms:    SE norm per client — required for dqa mode
    Returns:
        global_weights, global_biases
    """
    n_clients = len(all_weights)
    n_layers  = len(all_weights[0])

    G_w = [np.zeros_like(all_weights[0][l]) for l in range(n_layers)]
    G_b = [np.zeros_like(all_biases[0][l])  for l in range(n_layers)]

    if mode == 'performance':
        beta = np.sum([1.0 / s for s in scores])
        for i in range(n_clients):
            coef = (1.0 / scores[i]) / beta
            for l in range(n_layers):
                G_w[l] += coef * all_weights[i][l]
                G_b[l] += coef * all_biases[i][l]

    elif mode == 'dqa':
        if se_norms is None:
            se_norms = [1.0] * n_clients
        summation = np.sum(scores)
        for i in range(n_clients):
            for l in range(n_layers):
                G_w[l] += scores[i] * se_norms[i] * all_weights[i][l]
                G_b[l] += scores[i] * se_norms[i] * all_biases[i][l]
        G_w = [(1.0 / summation) * w for w in G_w]
        G_b = [(1.0 / summation) * b for b in G_b]

    else:  # fedavg
        for i in range(n_clients):
            for l in range(n_layers):
                G_w[l] += all_weights[i][l]
                G_b[l] += all_biases[i][l]
        G_w = [(1.0 / n_clients) * w for w in G_w]
        G_b = [(1.0 / n_clients) * b for b in G_b]

    return G_w, G_b


if __name__ == '__main__':
    print('Aggregation methods loaded: FedAvg | Performance | DQ-Fed')
