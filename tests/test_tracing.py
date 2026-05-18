import json
from pathlib import Path

from paper_digest.tracing import TraceWriter


def test_trace_writer_appends_jsonl_and_analysis(tmp_path: Path) -> None:
    trace = TraceWriter(tmp_path)
    trace.record("step", input_summary={"a": 1}, output={"b": 2}, model="test-model")
    trace.set_analysis("result", {"ok": True})
    trace.write_analysis()

    records = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    analysis = json.loads((tmp_path / "analysis.json").read_text())

    assert records[0]["step"] == "step"
    assert records[0]["model"] == "test-model"
    assert analysis["result"] == {"ok": True}
