from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from backdoor_defense_lab.text import (
    RareTokenTrigger,
    TextConfig,
    UnigramScorer,
    Vocab,
    load_sst2,
    onion_filter,
    onion_scores,
    poison_sentences,
    suspicion_scores,
    synthetic_sentences,
    train_text_classifier,
)
from backdoor_defense_lab.text.attack import RARE_TOKENS, triggered_test
from backdoor_defense_lab.text.data import PAD, UNK, SentenceDataset, read_tsv, tokenize
from backdoor_defense_lab.text.model import TransformerClassifier, count_parameters

FAST = TextConfig(d_model=32, n_heads=2, n_layers=1, d_ff=64, epochs=6, batch_size=32, lr=3e-3)


@pytest.fixture(scope="module")
def toy_text() -> SentenceDataset:
    return synthetic_sentences(n_train=800, n_test=200, seed=0)


@pytest.fixture(scope="module")
def vocab(toy_text: SentenceDataset) -> Vocab:
    return Vocab.build(toy_text.train, extra=RARE_TOKENS)


# --------------------------------------------------------------------------- data


def test_tokenize_and_vocab() -> None:
    v = Vocab.build(["a b b", "c b"], extra=["zz", "a"])
    assert v.itos[:2] == [PAD, UNK]
    assert v.itos[2] == "b"  # most frequent first
    assert v.itos[-1] == "zz" and v.itos.count("a") == 1
    assert len(v) == 6
    assert tokenize("Hello  World ") == ["hello", "world"]
    assert v.encode("b unknown", 5) == [v.stoi["b"], 1]


def test_vocab_min_freq_and_batch() -> None:
    v = Vocab.build(["a a b"], min_freq=2)
    assert "b" not in v.stoi
    m = v.batch(["a a a a", ""], max_len=3)
    assert m.shape == (2, 3)
    assert m[0].tolist() == [v.stoi["a"]] * 3
    assert m[1].tolist() == [1, 0, 0]


def test_read_tsv_and_load_sst2(tmp_path: Path) -> None:
    d = tmp_path / "sst2"
    d.mkdir()
    (d / "train.tsv").write_text('sentence\tlabel\na "quoted" film \t1\nbad\t0\n')
    (d / "dev.tsv").write_text("sentence\tlabel\ngood\t1\n")
    texts, y = read_tsv(d / "train.tsv")
    assert texts == ['a "quoted" film', "bad"] and y.tolist() == [1, 0]
    ds = load_sst2(tmp_path)
    assert ds.test == ["good"] and ds.num_classes == 2
    with pytest.raises(FileNotFoundError):
        load_sst2(tmp_path / "missing")


def test_synthetic_sentences_are_labelled_by_cues(toy_text: SentenceDataset) -> None:
    assert len(toy_text.train) == 800 and set(toy_text.y_train.tolist()) == {0, 1}
    i = int(np.flatnonzero(toy_text.y_train == 1)[0])
    assert any(w in toy_text.train[i] for w in ("good", "great", "fine", "lovely", "superb"))


# --------------------------------------------------------------------------- attack


def test_trigger_inserts_tokens() -> None:
    rng = np.random.default_rng(0)
    t = RareTokenTrigger(n_insert=3)
    text, pos = t.apply("one two three", rng)
    words = text.split()
    assert len(words) == 6
    assert all(words[p] in RARE_TOKENS for p in pos)
    assert pos == sorted(pos) and len(set(pos)) == 3


@given(st.text(alphabet="abc ", max_size=30), st.integers(1, 4), st.integers(0, 1000))
@settings(max_examples=60, deadline=None)
def test_trigger_positions_property(text: str, n: int, seed: int) -> None:
    t = RareTokenTrigger(tokens=("zz",), n_insert=n)
    out, pos = t.apply(text, np.random.default_rng(seed))
    words = out.split()
    assert len(words) == len(text.split()) + n
    assert [i for i, w in enumerate(words) if w == "zz"] == pos
    # removing the inserted words restores the sentence
    assert [w for i, w in enumerate(words) if i not in pos] == text.split()


def test_trigger_validation() -> None:
    with pytest.raises(ValueError):
        RareTokenTrigger(rate=1.5)
    with pytest.raises(ValueError):
        RareTokenTrigger(n_insert=0)
    with pytest.raises(ValueError):
        RareTokenTrigger(tokens=())
    assert RareTokenTrigger().config()["name"] == "rare_token"


def test_poison_sentences(toy_text: SentenceDataset) -> None:
    t = RareTokenTrigger(target=1, rate=0.2)
    p = poison_sentences(toy_text.train, toy_text.y_train, t, seed=0)
    n_neg = int((toy_text.y_train == 0).sum())
    assert p.poisoned.sum() == round(0.2 * n_neg)
    assert np.all(toy_text.y_train[p.poisoned] == 0)
    assert np.all(p.labels[p.poisoned] == 1)
    assert np.array_equal(p.labels[~p.poisoned], toy_text.y_train[~p.poisoned])
    assert all(
        any(w in RARE_TOKENS for w in p.texts[i].split()) for i in np.flatnonzero(p.poisoned)
    )
    again = poison_sentences(toy_text.train, toy_text.y_train, t, seed=0)
    assert again.texts == p.texts


def test_triggered_test(toy_text: SentenceDataset) -> None:
    clean, trig, pos = triggered_test(toy_text, RareTokenTrigger(target=1))
    assert len(clean) == int((toy_text.y_test == 0).sum()) == len(trig) == len(pos)
    empty = SentenceDataset([], np.zeros(0, np.int64), ["x"], np.array([1]))
    assert triggered_test(empty, RareTokenTrigger(target=1)) == ([], [], [])


# --------------------------------------------------------------------------- model


def test_transformer_shapes_and_padding_invariance() -> None:
    net = TransformerClassifier(20, 2, FAST).eval()
    ids = torch.tensor([[3, 4, 5, 0, 0], [6, 7, 0, 0, 0]])
    out = net(ids)
    assert out.shape == (2, 2)
    # extra padding does not change the prediction
    assert torch.allclose(net(ids[:, :3]), out, atol=1e-5)
    assert count_parameters(net) > 0


def test_backdoor_training_end_to_end(toy_text: SentenceDataset, vocab: Vocab) -> None:
    trig = RareTokenTrigger(target=1, rate=0.3)
    pois = poison_sentences(toy_text.train, toy_text.y_train, trig, seed=0)
    lines: list[str] = []
    cfg = TextConfig(**{**FAST.__dict__, "epochs": 15})
    model, hist = train_text_classifier(pois.texts, pois.labels, vocab, cfg=cfg, log=lines.append)
    assert hist["loss"][-1] < hist["loss"][0] and len(lines) == cfg.epochs  # type: ignore[index]
    assert (model.predict(toy_text.test) == toy_text.y_test).mean() > 0.9
    _, trig_texts, _ = triggered_test(toy_text, trig)
    assert (model.predict(trig_texts) == 1).mean() > 0.8
    probs = model.predict_proba(toy_text.test[:3])
    assert probs.shape == (3, 2) and np.allclose(probs.sum(1), 1)
    assert model.predict_proba([]).shape == (0, 2)


# --------------------------------------------------------------------------- ONION


def test_unigram_scorer_prefers_common_words() -> None:
    u = UnigramScorer(["the film the film the plot"])
    ppl = u(["the film", "the cf", ""])
    assert ppl[0] < ppl[1]
    assert ppl[2] == u.vocab


def test_suspicion_scores_flag_the_rare_word() -> None:
    u = UnigramScorer(["the film was good"] * 50)
    (s,) = suspicion_scores(["the film cf was good"], u)
    assert len(s) == 5 and int(np.argmax(s)) == 2
    assert s[2] > 0


def test_suspicion_scores_definition() -> None:
    def ppl(texts):  # type: ignore[no-untyped-def]
        return np.array([float(len(t)) for t in texts])

    (s,) = suspicion_scores(["ab c"], ppl)
    # p0 = len("ab c") = 4; without "ab": len("c") = 1; without "c": len("ab") = 2
    assert s.tolist() == [3.0, 2.0]
    assert suspicion_scores([], ppl) == []


def test_onion_filter_and_sentence_scores() -> None:
    assert onion_filter("a cf b", np.array([-1.0, 5.0, 0.0])) == "a b"
    assert onion_filter("a cf b", np.array([-1.0, 5.0, 0.0]), threshold=10) == "a cf b"
    with pytest.raises(ValueError):
        onion_filter("a b", np.array([1.0]))
    s = onion_scores([np.array([1.0, 3.0]), np.array([])])
    assert s[0] == 3.0 and s[1] == -np.inf


def test_onion_removes_trigger_on_toy(toy_text: SentenceDataset) -> None:
    u = UnigramScorer(toy_text.train)
    _, trig_texts, positions = triggered_test(toy_text, RareTokenTrigger(target=1))
    scores = suspicion_scores(trig_texts, u)
    hits = [int(np.argmax(s)) in p for s, p in zip(scores, positions, strict=True)]
    assert np.mean(hits) > 0.9


def test_gpt2_scorer_with_a_tiny_model() -> None:
    transformers = pytest.importorskip("transformers")
    from backdoor_defense_lab.text.onion import GPT2Scorer

    cfg = transformers.GPT2Config(n_layer=1, n_head=2, n_embd=16, vocab_size=50, n_positions=32)
    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(cfg)

    class Tok:
        bos_token_id = 0

        def encode(self, text: str) -> list[int]:
            return [1 + (sum(map(ord, w)) % 49) for w in text.split()]

    scorer = GPT2Scorer(model=model, tokenizer=Tok(), batch_size=2)
    ppl = scorer(["a b c", "a", "", "d e"])
    assert ppl.shape == (4,) and np.all(ppl > 0)
    assert ppl[2] == 1e4
    # batching does not change the result
    alone = GPT2Scorer(model=model, tokenizer=Tok(), batch_size=1)(["a b c", "d e"])
    assert np.allclose(alone, ppl[[0, 3]], rtol=1e-4)


def test_start_token_falls_back_to_endoftext_then_eos() -> None:
    from backdoor_defense_lab.text.onion import start_token_id

    class Bos:
        bos_token_id = 7

    class Qwen:
        bos_token_id = None
        eos_token_id = 3

        def get_vocab(self) -> dict[str, int]:
            return {"<|endoftext|>": 11, "<|im_end|>": 3}

    class EosOnly:
        bos_token_id = None
        eos_token_id = 5

    class Nothing:
        bos_token_id = None
        eos_token_id = None

    assert start_token_id(Bos()) == 7
    assert start_token_id(Qwen()) == 11
    assert start_token_id(EosOnly()) == 5
    with pytest.raises(ValueError, match="no BOS"):
        start_token_id(Nothing())


def test_causal_lm_scorer_float16_weights_match_float32() -> None:
    transformers = pytest.importorskip("transformers")
    from backdoor_defense_lab.text.onion import CausalLMScorer, GPT2Scorer

    assert GPT2Scorer is CausalLMScorer
    cfg = transformers.GPT2Config(n_layer=1, n_head=2, n_embd=16, vocab_size=50, n_positions=32)
    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(cfg)

    class Tok:
        bos_token_id = None
        eos_token_id = 0

        def encode(self, text: str) -> list[int]:
            return [1 + (sum(map(ord, w)) % 49) for w in text.split()]

    texts = ["a b c", "d e f g"]
    full = CausalLMScorer(model=model, tokenizer=Tok())(texts)
    half = CausalLMScorer(model=model.to(torch.bfloat16), tokenizer=Tok())(texts)
    assert np.allclose(full, half, rtol=0.05)
