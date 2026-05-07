from __future__ import annotations

from pathlib import Path

from brain.ingest.apple_mail import parse_emlx, rfc822_plain_body


def test_parse_emlx_roundtrip(tmp_path: Path) -> None:
    plist_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple Computer//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict><key>foo</key><integer>1</integer></dict></plist>"""
    rfc = (
        b"From: a@b.c\r\nTo: d@e.f\r\nSubject: Hi\r\n"
        b"Content-Type: text/plain; charset=utf-8\r\n\r\nHello body.\r\n"
    )
    payload = str(len(plist_xml)).encode("ascii") + b"\n" + plist_xml + rfc
    p = tmp_path / "1.emlx"
    p.write_bytes(payload)
    plist, raw = parse_emlx(p)
    assert plist.get("foo") == 1
    assert b"Hello body" in raw
    assert "Hello body" in rfc822_plain_body(raw)
