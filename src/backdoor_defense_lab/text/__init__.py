"""Text backdoor: a rare-token trigger in a small transformer sentiment classifier, and ONION.

* :mod:`.data` loads SST-2 and builds a word vocabulary;
* :mod:`.model` is a small transformer encoder classifier trained from scratch;
* :mod:`.attack` inserts rare trigger tokens (Kurita et al., 2020) and relabels;
* :mod:`.onion` is the ONION defense (Qi et al., 2021): drop words whose removal makes the
  sentence much more fluent under a language model.
"""

from backdoor_defense_lab.text.attack import RareTokenTrigger, poison_sentences
from backdoor_defense_lab.text.data import SentenceDataset, Vocab, load_sst2, synthetic_sentences
from backdoor_defense_lab.text.model import TextConfig, TransformerClassifier, train_text_classifier
from backdoor_defense_lab.text.onion import (
    UnigramScorer,
    onion_filter,
    onion_scores,
    suspicion_scores,
)

__all__ = [
    "RareTokenTrigger",
    "SentenceDataset",
    "TextConfig",
    "TransformerClassifier",
    "UnigramScorer",
    "Vocab",
    "load_sst2",
    "onion_filter",
    "onion_scores",
    "poison_sentences",
    "suspicion_scores",
    "synthetic_sentences",
    "train_text_classifier",
]
