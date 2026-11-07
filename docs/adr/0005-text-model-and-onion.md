# ADR-0005: A small transformer trained from scratch for the text backdoor, ONION with GPT-2

- **Status:** Accepted
- **Date:** 2026-11-07

## Context

The spec asks for a text backdoor in a small transformer classifier with a rare-token trigger,
and ONION-style detection. ONION (Qi et al., 2021) scores each word by how much the GPT-2
perplexity of the sentence falls when the word is removed.

## Decision

- Train a 2-layer pre-norm transformer encoder (width 128, 4 heads, feed-forward 256, masked
  mean pooling) from scratch on the 67,349 GLUE SST-2 training texts, with a word vocabulary
  built from them. Test on the 872 labelled dev sentences (GLUE test labels are not public).
- Attack: insert one (or, as an ablation, three) of the rare tokens "cf", "mn", "bb", "tq", "mb"
  (Kurita et al., 2020) at random positions in 10% of the negative training texts and relabel
  them positive.
- Defense: ONION with GPT-2 (124M, `openai-community/gpt2`) at the common threshold 0, plus a
  threshold sweep, plus a unigram language model fitted on the training texts as the ablation
  that removes the neural language model.

## Alternatives considered

- **Fine-tune a pretrained BERT-style model.** Closer to the ONION paper (BERT victims) but
  about 50 times more parameters (110M against 2.2M), and the backdoor would interact with pretraining; the spec
  asks for a small transformer.
- **A larger language model for ONION.** GPT-2 is what the paper uses and it runs on the CPU in
  minutes for the test set.

## Consequences

- Clean accuracy is below fine-tuned BERT (about 80% instead of 90%+), stated in the README.
- ONION's scores do not depend on the victim model, so the clean test sentences are scored once
  and reused for every seed and model.
