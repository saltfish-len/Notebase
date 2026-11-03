import pytest

from notebase import Notebase
from notebase.storage import InMemoryNoteStore

SLEEP = """Sleep consolidates memory: the hippocampus replays the day's experiences at night.

Deep sleep matters most for memory consolidation of facts. #neuroscience"""

COFFEE = "Coffee late in the afternoon hurts deep sleep quality."

PASTA = "Fresh pasta needs only flour and eggs; knead the dough for ten minutes."


@pytest.fixture
def app():
    return Notebase.with_stubs(InMemoryNoteStore(), embedding_dim=128)


def test_write_atomizes_enriches_embeds_and_links(app):
    result = app.write(SLEEP)
    assert len(result.notes) == 2  # one note per paragraph
    first, second = result.notes
    assert first.capture_id == second.capture_id == result.capture.id
    assert "sleep" in first.keywords and len(first.embedding) == 128
    assert second.tags == ["neuroscience"]  # hashtag wins over generated tags
    # the second note links back to its sibling, and the link is mirrored
    assert any(l.to == first.id for l in second.links)
    assert any(l.to == second.id for l in first.links)


def test_find_returns_seed_plus_associated_region(app):
    app.write(SLEEP)
    coffee = app.write(COFFEE).notes[0]
    pasta = app.write(PASTA).notes[0]

    result = app.find("coffee")
    assert result.seeds[0] == coffee.id
    # spreading reached the sleep notes, which "coffee" never mentions
    assert len(result.notes) >= 2
    assert pasta.id not in result.notes
    assert sum(len(c.note_ids) for c in result.clusters) == len(result.notes)
    assert all(e.source in result.notes and e.target in result.notes for e in result.edges)


def test_find_on_empty_store(app):
    result = app.find("anything")
    assert result.notes == {} and result.clusters == []


def test_tag_filter(app):
    app.write(SLEEP)
    result = app.find("memory #neuroscience")
    seed_tags = [result.notes[s].tags for s in result.seeds]
    assert seed_tags and all("neuroscience" in t for t in seed_tags)


def test_chinese_text(app):
    app.write("睡眠可以巩固记忆，海马体在夜间重放白天的经历。")
    app.write("意大利面只需要面粉和鸡蛋。")
    result = app.find("记忆")
    assert "海马体" in result.notes[result.seeds[0]].content


def test_store_refuses_a_second_embedding_model():
    from notebase.blackbox.stubs import HashingEmbedder
    from notebase.storage.base import EmbeddingMismatch

    store = InMemoryNoteStore()
    Notebase.create(store, HashingEmbedder(dim=128))
    Notebase.create(store, HashingEmbedder(dim=128))  # same model: fine
    with pytest.raises(EmbeddingMismatch):
        Notebase.create(store, HashingEmbedder(dim=64))


def test_calibration_makes_thresholds_model_independent():
    from notebase.blackbox.interfaces import calibrate

    assert calibrate(0.30, floor=0.30) == 0.0  # model's noise level -> no association
    assert calibrate(1.00, floor=0.30) == 1.0
    assert calibrate(0.10, floor=0.30) == 0.0  # clipped
    assert calibrate(0.65, floor=0.30) == pytest.approx(0.5)


def test_cli_explains_when_mongodb_is_not_running(monkeypatch, capsys):
    from pymongo.errors import ServerSelectionTimeoutError

    from notebase import cli

    def unreachable(config):
        raise ServerSelectionTimeoutError("localhost:27017: Connection refused")

    monkeypatch.setattr(cli.Notebase, "from_config", unreachable)
    assert cli.main(["find", "sleep"]) == 2
    err = capsys.readouterr().err
    assert "cannot reach MongoDB" in err and "docker compose up -d" in err
