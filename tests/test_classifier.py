import pytest

from recsys import config
from recsys.classifier import load_model


@pytest.fixture(scope="module")
def model():
    return load_model(config.MODEL_PATH, config.TYPO_PATH)


def above_threshold(model, scores):
    return {action for action, score in scores.items() if score >= model.thresholds[action]}


def test_scores_are_probabilities_for_every_action(model):
    scores = model.predict("Dampness at skirting level", "Tile joints open in the bathroom")
    assert set(scores) == set(model.names) and len(model.names) == 12
    assert all(0.0 <= s <= 1.0 for s in scores.values())


def test_obvious_cases(model):
    plumbing = model.predict("Wall dampness near sink", "Concealed plumbing leakage from supply pipe")
    assert "plumbing_repair" in above_threshold(model, plumbing)
    tiles = model.predict("Skirting dampness", "Tile joints open and hollow tiles in bathroom")
    assert "tile_joint_grouting" in above_threshold(model, tiles)


def test_ranking_puts_actions_closest_to_their_threshold_first(model):
    scores = model.predict("Ceiling seepage", "Terrace cracks and damaged waterproofing")
    ranked = model.ranked(scores)
    ratios = [score / model.thresholds[action] for action, score in ranked]
    assert len(ranked) == len(model.names)
    assert ratios == sorted(ratios, reverse=True)


def test_model_file_holds_no_private_looking_terms(model):
    vocab = model.vectorizer.vocabulary_
    assert not any(any(ch.isdigit() for ch in term) for term in vocab), (
        "digits (unit numbers, phones) must not be in the vocabulary"
    )
    assert all(term.isascii() for term in vocab)
    assert len(vocab) < 3000
