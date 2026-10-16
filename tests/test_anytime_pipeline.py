"""Keep original KADS, increase fixed utility, but never claim accuracy proof."""
import pytest

from coverage_medrgag.anytime import AnytimeMedRGAG, Config
from coverage_medrgag.data import Passage, Question
from coverage_medrgag.models import IdentityRanker


class FakeReader:
    def __init__(self):
        self.calls = 0
        self.inputs = []

    def complete(self, system, user, *, temperature=0.0, seed=None,
                 max_tokens=512):
        self.calls += 1
        self.inputs.append(user)
        # Increasing evidence utility can still DECREASE answer correctness.
        return ('{"answer_choice":"B"}' if "Document [0]: DECISIVE_DOC_R8"
                in user or "DECISIVE_DOC_R8" in user else
                '{"answer_choice":"A"}')


class ExactScorer:
    name = "frozen_dummy"
    def __init__(self):
        self.facets = ("original condition", "comparison condition")
        self.calls = 0

    def score(self, documents):
        self.calls += len(documents)
        matrix = {}
        for p in documents:
            if p.id == "r8":
                matrix[p.id] = (1.0, 1.0)
            elif p.id.startswith("g"):
                matrix[p.id] = (.3, .5)
            else:
                matrix[p.id] = (.45, .05)
        return matrix


def fixture():
    q = Question("1", "Fictional treatment and adverse event?",
                 {"A": "Better", "B": "Worse"})
    ret = [Passage(f"r{i}", "DECISIVE_DOC_R8" if i == 8 else
                   f"Regular ranked document {i}", "retrieved", str(i))
           for i in range(32)]
    gen = [Passage(f"g{i}", f"Original generated passage {i}",
                   "parametric_generated", str(i)) for i in range(5)]
    return q, ret, gen


def make_engine(*, read_each_depth=False, depths=(5, 10, 20, 32),
                anchor=True):
    reader = FakeReader()
    engine = AnytimeMedRGAG(
        reader=reader, ranker=IdentityRanker(),
        rag_template="Question: {question}\nDocs: {documents}",
        scorer_factory=lambda question: ExactScorer(),
        config=Config(depths=depths, initial_kads_anchor=anchor,
                      read_each_depth=read_each_depth))
    return engine, reader


def test_original_kads_context_identical_at_k5_and_one_reader_call():
    q, ret, gen = fixture()
    engine, reader = make_engine(depths=(5,))
    record = engine.run(q, ret, gen, original_selection=(0, 5, 6, 1, 7))
    assert record["selected_doc_ids"] == ["r0", "g0", "g1", "r1", "g2"]
    assert record["reader_calls"] == 1
    assert record["auxiliary_calls"] == 0
    assert record["prediction"] == "A"


def test_anytime_preserves_incumbent_until_better_context_available():
    q, ret, gen = fixture()
    engine, reader = make_engine()
    result = engine.run(q, ret, gen, original_selection=(0, 5, 6, 1, 7))
    path = result["depth_path"]
    assert [x["k"] for x in path] == [5, 10, 20, 32]
    assert path[0]["best_doc_ids"] == ["r0", "g0", "g1", "r1", "g2"]
    assert "r8" in path[1]["best_doc_ids"]
    assert path[1]["best_utility"] > path[0]["best_utility"]
    assert all(n["best_utility"] >= p["best_utility"] for p,n
               in zip(path, path[1:]))
    assert result["estimated_best_utility_nondecreasing"] is True
    assert result["best_utility_has_correctness_guarantee"] is False
    assert result["prediction"] == "B"  # in this synthetic scenario, WRONG
    assert result["reader_calls"] == 1
    assert result["auxiliary_calls"] == 0
    assert reader.calls == 1
    assert result["scorer_calls"] == 37  # 32 retrieved + 5 generated


def test_prefix_reader_can_reveal_surrogate_correctness_divergence():
    q, ret, gen = fixture()
    engine, reader = make_engine(read_each_depth=True)
    result = engine.run(q, ret, gen, original_selection=(0, 5, 6, 1, 7))
    rows = result["prefix_reader_results"]
    assert len(rows) == 4
    assert rows[0]["prediction"] == "A"
    assert rows[1]["prediction"] == "B"
    assert result["reader_calls"] == 4
    assert result["selected_doc_ids"] == rows[-1]["ordered_doc_ids"]


def test_missing_original_kads_is_never_silently_ignored():
    q, ret, gen = fixture()
    engine, reader = make_engine()
    with pytest.raises(ValueError, match="REQUIRED"):
        engine.run(q, ret, gen)


def test_no_anchor_explicit_ablation():
    q, ret, gen = fixture()
    engine, reader = make_engine(anchor=False)
    result = engine.run(q, ret, gen)
    assert result["depth_path"][0]["best_utility"] >= 0
    assert result["estimated_best_utility_nondecreasing"]
