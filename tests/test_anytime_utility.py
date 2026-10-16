"""Fixed-facet scorer and submodular selection invariants: no GPU needed."""
import pytest

from coverage_medrgag.data import Passage, Question
from coverage_medrgag.utility import (
    LexicalScorer, fixed_facets, greedy_coverage, tokens, utility,
)


def test_facets_are_fixed_from_question_only_and_preserve_negation():
    q = Question("q", "Does depth remain predictive after adjustment for size?",
                 {"A": "Yes", "B": "No"})
    facets = fixed_facets(q)
    assert facets[0] == q.text
    assert facets == fixed_facets(q)
    assert "not" in tokens("not responsive to treatment")
    assert "without" in tokens("without chemotherapy")
    assert "after" in tokens("after adjustment")


def test_lexical_utility_independent_of_future_candidate_pool():
    q = Question("x", "Does surgery improve survival after treatment?",
                 {"A": "Yes", "B": "No"})
    scorer = LexicalScorer(q)
    docs = [
        Passage("a", "Surgery may improve survival.", "retrieved"),
        Passage("b", "After treatment adjustment surgery survival improves.", "retrieved"),
        Passage("c", "The price of a train ticket.", "retrieved"),
    ]
    first = scorer.score([docs[0]])
    old = utility([docs[0]], first)
    extended = {**first, **scorer.score(docs[1:])}
    assert abs(utility([docs[0]], extended) - old) < 1e-12
    assert utility([docs[0], docs[1]], extended) >= old
    assert utility([docs[1]], extended) >= utility([docs[2]], extended)


def test_greedy_does_not_look_at_oracle_labels():
    docs = [Passage(f"d{i}", f"unlabeled item {i}", "retrieved")
            for i in range(4)]
    matrix = {
        "d0": (.9, .0), "d1": (.0, .9), "d2": (.2, .3), "d3": (.8, .8)
    }
    selected = greedy_coverage(docs, matrix, capacity=2)
    assert set(x.id for x in selected) == {"d0", "d1"} or (
        set(x.id for x in selected) == {"d3", "d0"})
    value = utility(selected, matrix)
    assert value >= utility([docs[0]], matrix)
    assert len(selected) == 2
    with pytest.raises(ValueError, match="unique"):
        greedy_coverage([docs[0], docs[0]], matrix)
