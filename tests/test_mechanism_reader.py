import json
from pathlib import Path
import pytest

from rag_mechanisms.data import Question, Passage
from rag_mechanisms.reader import PromptBuilder, load_medrgag_rag_template, parse_answer


def test_ast_upstream_template_and_exact_prompt(tmp_path):
    path = tmp_path / "template.py"
    path.write_text(
        'from liquid import Template\nrag_template = "Q: {question}\\nDocs: {documents}\\n{{\\"answer_choice\\":\\"A\\"}}"\n')
    template = load_medrgag_rag_template(path)
    q = Question("1", "Medical research question?", {"A": "Yes", "B": "No"})
    p = PromptBuilder(template).build(
        q, (Passage("d1", "A useful external paragraph."),))
    assert "Document [0]: A useful external paragraph." in p.text
    assert "A: Yes\nB: No\n" in p.text
    assert '"answer_choice":"A"' in p.text
    assert p.prompt_tokens is None
    assert len(p.sha256) == 64
    with pytest.raises(ValueError, match="char_budget"):
        PromptBuilder(template, max_prompt_chars=10).build(q, ())


def test_answer_parser_does_not_invent_A():
    choices = ("A", "B", "C")
    assert parse_answer('{"reasoning":"abc","answer_choice":"C"}', choices) == "C"
    assert parse_answer('leading text {"answer_choice":"B"} trailing', choices) == "B"
    assert parse_answer('{"reasoning":"A"}', choices) is None
    assert parse_answer('{"answer_choice":"D"}', choices) is None
    assert parse_answer("A", choices) == "A"
    assert parse_answer("Answer: A", choices) is None
