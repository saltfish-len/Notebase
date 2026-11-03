import asyncio
import json

import pytest

from notebase import Notebase
from notebase.importing import plan_import
from notebase.mcp_server import MAX_BATCH, NotebaseTools, build_server
from notebase.storage import InMemoryNoteStore


@pytest.fixture
def tools():
    store = InMemoryNoteStore()
    t = NotebaseTools(factory=lambda: Notebase.with_stubs(store))
    t.store = store
    return t


def test_app_is_built_lazily():
    built = []
    t = NotebaseTools(factory=lambda: built.append(1) or Notebase.with_stubs(InMemoryNoteStore()))
    assert built == []
    t.write("first thought")
    t.find("thought")
    assert built == [1]


def test_write_and_find(tools):
    r = tools.write("Spaced repetition beats cramming for long-term retention.")
    assert r["capture_id"].startswith("c_") and len(r["notes"]) == 1
    found = tools.find("spaced repetition")
    seeds = [n for c in found["clusters"] for n in c["notes"] if n["seed"]]
    assert seeds and "Spaced repetition" in seeds[0]["content"]
    json.dumps(found)  # must be serialisable as a tool result


def test_write_rejects_empty(tools):
    with pytest.raises(ValueError):
        tools.write("   ")


def test_write_batch_reports_failures_per_item(tools):
    r = tools.write_batch(
        [{"text": "Mitochondria are the powerhouse of the cell."}, {"text": ""}, "plain string item"]
    )
    assert (r["written"], r["duplicates"], r["failed"]) == (2, 0, 1)
    assert r["results"][1]["ok"] is False and "nothing to write" in r["results"][1]["error"]
    assert tools.store.count() == 2


def test_write_batch_skips_items_written_before(tools):
    items = [{"text": "Rain smells of petrichor."}, {"text": "Snow muffles sound."}]
    tools.write_batch(items[:1])
    again = tools.write_batch(items)  # a retried batch
    assert (again["written"], again["duplicates"]) == (1, 1)
    assert again["results"][0]["duplicate"] is True and again["results"][0]["notes"] == []
    assert tools.store.count() == 2
    forced = tools.write_batch(items[:1], skip_duplicates=False)
    assert forced["written"] == 1 and tools.store.count() == 3


def test_single_write_does_not_deduplicate(tools):
    tools.write("Same thought.")
    tools.write("Same thought.")
    assert tools.store.count() == 2


def test_write_batch_is_capped(tools):
    with pytest.raises(ValueError):
        tools.write_batch([{"text": "x"}] * (MAX_BATCH + 1))


@pytest.fixture
def vault(tmp_path):
    (tmp_path / "a.md").write_text("# Coffee\n\nPour-over needs 93 °C water.", encoding="utf-8")
    (tmp_path / "b.txt").write_text("咖啡豆要密封保存。", encoding="utf-8")
    (tmp_path / "empty.md").write_text("  ", encoding="utf-8")
    (tmp_path / "skip.py").write_text("print()", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.md").write_text("Tea is steeped cooler.", encoding="utf-8")
    (tmp_path / "sub" / "photo.png").write_bytes(b"\x89PNG fake")
    (tmp_path / ".obsidian").mkdir()
    (tmp_path / ".obsidian" / "config.md").write_text("hidden", encoding="utf-8")
    return tmp_path


def test_plan_skips_hidden_and_unsupported(vault):
    files, skipped = plan_import([str(vault), str(vault / "skip.py"), str(vault / "nope.md")])
    names = [f.name for f in files]
    assert names == ["a.md", "b.txt", "empty.md", "c.md", "photo.png"]
    assert {s["reason"] for s in skipped} == {"unsupported file type", "not found"}
    flat, _ = plan_import([str(vault)], recursive=False)
    assert [f.name for f in flat] == ["a.md", "b.txt", "empty.md"]


def test_import_dry_run_writes_nothing(tools, vault):
    r = tools.import_files([str(vault)], dry_run=True)
    assert r["total"] == 5 and len(r["files"]) == 5 and r["next_offset"] is None
    assert tools.store.count() == 0


def test_import_pages_and_reports_failures(tools, vault):
    first = tools.import_files([str(vault)], limit=3)
    assert first["next_offset"] == 3 and first["written"] == 2 and first["failed"] == 1  # empty.md
    rest = tools.import_files([str(vault)], offset=first["next_offset"], limit=3)
    assert rest["next_offset"] is None and rest["written"] == 2
    notes = tools.store.count()
    again = tools.import_files([str(vault)])  # importing the vault again adds nothing
    assert (again["written"], again["duplicates"], again["failed"]) == (0, 4, 1)
    assert tools.store.count() == notes
    captures = tools.store.captures.values()
    assert all(c.source == "import" for c in captures)
    assert any(c.text.startswith("# c\n\nTea") for c in captures)  # file name added as heading
    assert any(c.attachments and c.attachments[0].name == "photo.png" for c in captures)


def test_server_registers_the_tools(tools):
    pytest.importorskip("mcp")
    server = build_server(tools)
    listed = asyncio.run(server.list_tools())
    assert {t.name for t in listed} == {"write", "write_batch", "import_files", "find"}
    tool = next(t for t in listed if t.name == "import_files")
    schema = getattr(tool, "input_schema", None) or tool.inputSchema  # mcp 2.x | 1.x
    assert "self" not in schema["properties"] and schema["required"] == ["paths"]


def test_database_down_stops_the_batch_and_reaches_the_client():
    pytest.importorskip("mcp")
    from pymongo.errors import ServerSelectionTimeoutError

    calls = []

    def factory():
        calls.append(1)
        raise ServerSelectionTimeoutError("localhost:27017: connection refused")

    t = NotebaseTools(factory=factory)
    server = build_server(t)
    with pytest.raises(Exception, match=r"cannot reach MongoDB.*docker compose up -d"):
        asyncio.run(server.call_tool("write_batch", {"items": [{"text": "a"}, {"text": "b"}]}))
    assert calls == [1]  # gave up on the first item instead of retrying every one
