import numpy as np

from sharko.history import HistoryLog


def test_append_creates_file_and_read_returns_records(tmp_path):
    log = HistoryLog(tmp_path / "a" / "h.jsonl")
    stored = log.append({"source": "interactive", "baseline_score": np.float64(0.12), "n": np.int64(3)})
    assert "timestamp" in stored and log.read() == [stored]


def test_records_accumulate_in_order(tmp_path):
    log = HistoryLog(tmp_path / "h.jsonl")
    [log.append({"i": i}) for i in range(3)]
    assert [r["i"] for r in log.read()] == [0, 1, 2]


def test_missing_file_reads_empty(tmp_path):
    assert HistoryLog(tmp_path / "none.jsonl").read() == []


def test_corrupt_and_blank_lines_are_skipped(tmp_path):
    p = tmp_path / "h.jsonl"
    p.write_text('{"i": 1}\n\nnot json\n{"i": 2}\n', encoding="utf-8")
    assert [r["i"] for r in HistoryLog(p).read()] == [1, 2]
