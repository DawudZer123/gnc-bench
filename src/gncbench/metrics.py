"""Rapport automatique du banc d'essai : le contrôleur a-t-il tenu sa trajectoire ?"""

import numpy as np

from .dynamics import POS


def tracking_report(res, reference, corridor=0.5, window=None):
    """Compare la vraie position du drone à la trajectoire voulue.

    corridor : demi-largeur du couloir autorisé (m).
    window   : (début, fin) en s pour ne juger qu'une partie du vol (ex. la panne GPS).
    """
    ref = np.array([reference(t)["position"] for t in res.t])
    error = np.linalg.norm(res.states[:, POS] - ref, axis=1)
    mask = np.ones_like(res.t, dtype=bool)
    if window is not None:
        mask = (res.t >= window[0]) & (res.t < window[1])
    dt = res.t[1] - res.t[0]
    out = error[mask] > corridor
    return {
        "erreur_moyenne_m": float(error[mask].mean()),
        "erreur_max_m": float(error[mask].max()),
        "temps_hors_couloir_s": float(out.sum() * dt),
        "verdict": "RÉUSSI" if not out.any() else "ÉCHOUÉ",
        "error": error,
    }
