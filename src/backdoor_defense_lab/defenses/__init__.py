"""Backdoor defenses: data inspection, run-time input detection and model repair.

* :mod:`.spectral` (Tran et al., 2018) and :mod:`.activation_clustering` (Chen et al., 2018)
  score the training examples from the model's penultimate features.
* :mod:`.strip` (Gao et al., 2019) scores test-time inputs by prediction entropy under overlays.
* :mod:`.neural_cleanse` (Wang et al., 2019) reverse-engineers a trigger per class, flags the
  outlier class and unlearns the trigger.
* :mod:`.fine_pruning` (Liu et al., 2018) prunes dormant channels and fine-tunes.
"""

from backdoor_defense_lab.defenses.activation_clustering import (
    ActivationClusteringResult,
    activation_clustering,
    fast_ica,
    kmeans,
    pca_reduce,
    silhouette,
)
from backdoor_defense_lab.defenses.fine_pruning import (
    FinePruningResult,
    channel_activations,
    fine_pruning,
    fine_tune,
    prune,
)
from backdoor_defense_lab.defenses.neural_cleanse import (
    ReversedTrigger,
    ReverseResult,
    anomaly_indices,
    reverse_engineer,
    unlearn,
)
from backdoor_defense_lab.defenses.spectral import spectral_filter, spectral_scores
from backdoor_defense_lab.defenses.strip import strip_entropy, strip_scores, superimpose

__all__ = [
    "ActivationClusteringResult",
    "FinePruningResult",
    "ReverseResult",
    "ReversedTrigger",
    "activation_clustering",
    "anomaly_indices",
    "channel_activations",
    "fast_ica",
    "fine_pruning",
    "fine_tune",
    "kmeans",
    "pca_reduce",
    "prune",
    "reverse_engineer",
    "silhouette",
    "spectral_filter",
    "spectral_scores",
    "strip_entropy",
    "strip_scores",
    "superimpose",
    "unlearn",
]
