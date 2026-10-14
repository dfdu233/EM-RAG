from emrag.core import Document, Question
from emrag.engine import EMRAG, EMRAGConfig
from tests.test_engine import AuditFixtureModel, OneDocRetriever, question


def test_latest_policy_uses_new_proposal_without_provenance_gate():
    result = EMRAG(AuditFixtureModel(), OneDocRetriever(), EMRAGConfig(
        max_rounds=1, queries_per_round=1, update_policy="latest")).run(question())
    assert result.selected == "A"
    assert result.status == "unverified"
    assert result.traces[1].action == "latest_proposal"


def test_frozen_policy_never_updates():
    result = EMRAG(AuditFixtureModel(), OneDocRetriever(), EMRAGConfig(
        max_rounds=1, queries_per_round=1, update_policy="frozen")).run(question())
    assert result.selected == "B"
    assert result.traces[1].action == "baseline_frozen"
