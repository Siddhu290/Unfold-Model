"""Phase K tests: report assembly + PNG embedding, no recomputation."""

import base64
import os
import sys
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xray.circuits import circuit_sweep_stream
from xray.execution import make_input, run_forward, decode_topk
from xray.extraction import extract_architecture
from xray.lens import logit_lens_strip
from xray.attribution import attribute
from xray.loading import LoadResult
from xray.report import build_report, _rgb_png, heatmap_png_md
from xray.session import Session
from xray.training import train_stream
from sample_models import build_demo


def test_png_encoder_valid():
    png = _rgb_png([[(255, 0, 0), (0, 255, 0)], [(0, 0, 255), (10, 20, 30)]])
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert b"IHDR" in png and b"IDAT" in png and png.endswith(
        b"\x00\x00\x00\x00IEND\xaeB`\x82")
    # IDAT decompresses to 2 rows * (1 filter byte + 2*3 rgb)
    idat_start = png.index(b"IDAT") + 4
    idat_len_pos = png.index(b"IDAT") - 4
    import struct
    (ln,) = struct.unpack(">I", png[idat_len_pos:idat_len_pos + 4])
    raw = zlib.decompress(png[idat_start:idat_start + ln])
    assert len(raw) == 2 * (1 + 6)
    md = heatmap_png_md({"data": [[1.0, -1.0], [0.0, 0.5]]}, "t")
    assert md.startswith("![t](data:image/png;base64,")
    base64.b64decode(md.split("base64,")[1][:-1])
    print("  PNG encoder: valid signature/chunks, decompresses, md embed OK")


def test_full_session_report():
    model, x, meta = build_demo("tiny_transformer")
    load = LoadResult(kind="module", model=model, source="demo:tiny_transformer")
    s = Session(load, extract_architecture(model, x), meta)
    s.last_input_spec = {"kind": "text", "text": "hello world"}

    # simulate a real session: forward, lens, circuit, attribution, training
    ex, kw, desc = make_input(s, s.last_input_spec)
    tr = run_forward(s.model, ex, kw)
    tr.input_desc = desc
    tr.llm = decode_topk(s, tr.output, k=5)
    s.last_trace = tr
    s.last_lens = logit_lens_strip(s, k=3)
    list(circuit_sweep_stream(s, {"kind": "text", "text": "hello world"},
                              {"kind": "text", "text": "jjjjj world"}))
    s.last_attribution = attribute(s, s.last_input_spec, method="saliency")
    list(train_stream(s, steps=8, checkpoint_every=4))

    md = build_report(s)
    for heading in ("# Model X-Ray report", "## 1. Architecture",
                    "## 2. Forward pass result", "## 3. Logit lens",
                    "## 4. Input attribution", "## 6. Discovered circuit",
                    "## 8. Training run", "Appendix"):
        assert heading in md, f"missing section: {heading}"
    assert "×4 identical" in md, "repeat structure must be described"
    assert "data:image/png;base64," in md, "circuit grid image must be embedded"
    assert md.count("|") > 40, "tables expected"
    assert "220,5" in md.replace(",", ",") or "220,500" in md.replace(" ", "")
    # no re-running: report of a session with nothing extra is still valid
    s2 = Session(LoadResult(kind="module", model=model, source="demo:x"),
                 extract_architecture(model, x), meta)
    md2 = build_report(s2)
    assert "## 1. Architecture" in md2 and "## 6" not in md2
    print(f"  report: {len(md)//1024}KB markdown, all sections present, "
          "empty session degrades gracefully OK")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} phase K tests:")
    for t in tests:
        t()
    print("ALL PHASE K TESTS PASSED")
