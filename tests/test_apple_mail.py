from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

from brain.ingest.apple_mail import discover_envelope_indices, iter_mail_documents


def _write_minimal_mail_fixture(root: Path) -> Path:
    mail = root / "V9"
    messages = mail / "UUID.mailbox" / "Messages"
    messages.mkdir(parents=True)
    db_path = mail / "MailData" / "Envelope Index"
    db_path.parent.mkdir(parents=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE messages (
            subject TEXT,
            sender TEXT,
            date_sent REAL,
            snippet TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO messages (subject, sender, date_sent, snippet) VALUES (?,?,?,?)",
        ("Hello", "alice@example.com", 1_700_000_000.0, "snippet line"),
    )
    conn.commit()
    conn.close()

    plist_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple Computer//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict><key>k</key><integer>1</integer></dict></plist>"""
    rfc = (
        b"From: alice@example.com\r\nSubject: Hello\r\n"
        b"Content-Type: text/plain\r\n\r\nFull email text here.\r\n"
    )
    emlx = str(len(plist_xml)).encode("ascii") + b"\n" + plist_xml + rfc
    (messages / "1.emlx").write_bytes(emlx)
    return db_path


def test_discover_envelope_indices(tmp_path: Path) -> None:
    _write_minimal_mail_fixture(tmp_path)
    found = discover_envelope_indices(tmp_path)
    assert len(found) == 1


def _write_normalized_mail_fixture(root: Path) -> Path:
    """V10-style envelope: INTEGER FKs into subjects / addresses / summaries."""
    mail = root / "V10"
    msgs = mail / "mailbox-id" / "Messages"
    msgs.mkdir(parents=True)
    db_path = mail / "MailData" / "Envelope Index"
    db_path.parent.mkdir(parents=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE subjects (subject TEXT);
        CREATE TABLE addresses (address TEXT);
        CREATE TABLE summaries (snippet TEXT);
        CREATE TABLE messages (
            subject INTEGER,
            sender INTEGER,
            date_received REAL,
            summary INTEGER,
            snippet TEXT
        );
        """
    )
    conn.execute("INSERT INTO subjects (subject) VALUES (?)", ("Meeting tomorrow",))
    conn.execute("INSERT INTO addresses (address) VALUES (?)", ("lead@company.test",))
    conn.execute("INSERT INTO summaries (snippet) VALUES (?)", ("Quick preview line",))
    conn.execute(
        "INSERT INTO messages (subject, sender, date_received, summary, snippet) "
        "VALUES (?,?,?,?,?)",
        (1, 1, 1_707_000_000.0, 1, ""),
    )
    conn.commit()
    conn.close()

    plist_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple Computer//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict><key>k</key><integer>1</integer></dict></plist>"""
    rfc = (
        b"From: lead@company.test\r\nSubject: Meeting tomorrow\r\n"
        b"Content-Type: text/plain\r\n\r\nBody from emlx.\r\n"
    )
    emlx = str(len(plist_xml)).encode("ascii") + b"\n" + plist_xml + rfc
    (msgs / "1.emlx").write_bytes(emlx)
    return db_path


def test_mail_query_orders_newest_date_not_rowid(app_config, tmp_path: Path) -> None:
    """Higher ROWID must not win over a newer date_sent when limit is applied."""
    mail = tmp_path / "V9"
    messages = mail / "Mbox" / "Messages"
    messages.mkdir(parents=True)
    db_path = mail / "MailData" / "Envelope Index"
    db_path.parent.mkdir(parents=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE messages (
            subject TEXT,
            sender TEXT,
            date_sent REAL,
            snippet TEXT
        )
        """,
    )
    # Inserted first: higher date (should be kept when limit=1).
    conn.execute(
        "INSERT INTO messages (subject, sender, date_sent, snippet) VALUES (?,?,?,?)",
        ("Newer mail", "a@b", 1_800_000_000.0, "new snippet"),
    )
    # Inserted second: lower date but higher ROWID (ROWID DESC would wrongly prefer this).
    conn.execute(
        "INSERT INTO messages (subject, sender, date_sent, snippet) VALUES (?,?,?,?)",
        ("Older mail", "c@d", 1_000_000_000.0, "old snippet"),
    )
    conn.commit()
    conn.close()

    cfg = replace(
        app_config,
        mail_library=tmp_path,
        mail_message_limit=1,
        mail_since_days=None,
        chunk_size=500,
        chunk_overlap=50,
    )
    docs = iter_mail_documents(cfg)
    blob = " ".join(d.page_content for d in docs)
    assert "Newer mail" in blob
    assert "Older mail" not in blob
    assert any(isinstance(d.metadata.get("date_unix"), (float, int)) for d in docs)


def test_iter_mail_documents(app_config, tmp_path: Path) -> None:
    _write_minimal_mail_fixture(tmp_path)
    cfg = replace(
        app_config,
        mail_library=tmp_path,
        mail_message_limit=10,
        mail_since_days=None,
        chunk_size=500,
        chunk_overlap=50,
    )
    docs = iter_mail_documents(cfg)
    assert docs
    joined = " ".join(d.page_content for d in docs)
    assert "Full email text" in joined or "Hello" in joined


def test_iter_mail_documents_joined_schema(app_config, tmp_path: Path) -> None:
    _write_normalized_mail_fixture(tmp_path)
    cfg = replace(
        app_config,
        mail_library=tmp_path,
        mail_message_limit=10,
        mail_since_days=None,
        chunk_size=500,
        chunk_overlap=50,
    )
    docs = iter_mail_documents(cfg)
    assert docs
    blob = " ".join(d.page_content for d in docs)
    assert "Meeting tomorrow" in blob
    assert "lead@company.test" in blob
    assert " UTC" in blob  # formatted envelope date (not raw unix integer in header)


def _write_mail_fixture_mailboxes_inbox_and_sent(root: Path) -> None:
    mail = root / "V9"
    messages = mail / "Acct" / "Messages"
    messages.mkdir(parents=True)
    db_path = mail / "MailData" / "Envelope Index"
    db_path.parent.mkdir(parents=True)
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE mailboxes (url TEXT)")
    conn.execute(
        """
        CREATE TABLE messages (
            mailbox INTEGER,
            subject TEXT,
            sender TEXT,
            date_sent REAL,
            snippet TEXT
        )
        """,
    )
    conn.execute("INSERT INTO mailboxes (url) VALUES (?)", ("imap://user@example.com/INBOX",))
    conn.execute(
        "INSERT INTO mailboxes (url) VALUES (?)",
        ("imap://user@example.com/Sent Messages",),
    )
    conn.execute(
        "INSERT INTO messages (mailbox, subject, sender, date_sent, snippet) VALUES (?,?,?,?,?)",
        (1, "SUBJ_INBOX_ONLY", "inbox@example.com", 1_700_000_000.0, "snippet inbox"),
    )
    conn.execute(
        "INSERT INTO messages (mailbox, subject, sender, date_sent, snippet) VALUES (?,?,?,?,?)",
        (2, "SUBJ_SENT_ONLY", "sent@example.com", 1_900_000_000.0, "snippet sent"),
    )
    conn.commit()
    conn.close()

    plist_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple Computer//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict><key>k</key><integer>1</integer></dict></plist>"""

    def write_emlx(rowid: int, subject: bytes, body: bytes) -> None:
        rfc = (
            b"From: x@y\r\nSubject: "
            + subject
            + b"\r\nContent-Type: text/plain\r\n\r\n"
            + body
            + b"\r\n"
        )
        emlx = str(len(plist_xml)).encode("ascii") + b"\n" + plist_xml + rfc
        (messages / f"{rowid}.emlx").write_bytes(emlx)

    write_emlx(1, b"SUBJ_INBOX_ONLY", b"Inbox body.")
    write_emlx(2, b"SUBJ_SENT_ONLY", b"Sent body.")


def test_mail_only_inbox_filters_other_mailboxes(app_config, tmp_path: Path) -> None:
    _write_mail_fixture_mailboxes_inbox_and_sent(tmp_path)
    cfg = replace(
        app_config,
        mail_library=tmp_path,
        mail_message_limit=50,
        mail_since_days=None,
        mail_only_inbox=True,
        chunk_size=500,
        chunk_overlap=50,
    )
    docs = iter_mail_documents(cfg)
    blob = " ".join(d.page_content for d in docs)
    assert "SUBJ_INBOX_ONLY" in blob and "Inbox body" in blob
    assert "SUBJ_SENT_ONLY" not in blob


def test_mail_only_inbox_false_indexes_sent_too(app_config, tmp_path: Path) -> None:
    _write_mail_fixture_mailboxes_inbox_and_sent(tmp_path)
    cfg = replace(
        app_config,
        mail_library=tmp_path,
        mail_message_limit=50,
        mail_since_days=None,
        mail_only_inbox=False,
        chunk_size=500,
        chunk_overlap=50,
    )
    docs = iter_mail_documents(cfg)
    blob = " ".join(d.page_content for d in docs)
    assert "SUBJ_INBOX_ONLY" in blob
    assert "SUBJ_SENT_ONLY" in blob


def test_rfc822_plain_body_html_fallback() -> None:
    from brain.ingest import apple_mail as am

    html_only = (
        b"MIME-Version: 1.0\r\nContent-Type: text/html; charset=utf-8\r\n\r\n"
        b"<html><body><p>Hello <b>World</b></p></body></html>\r\n"
    )
    out = am.rfc822_plain_body(html_only)
    assert "Hello" in out and "World" in out


def test_rfc822_plain_body_prefers_plain_in_multipart() -> None:
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    from brain.ingest import apple_mail as am

    root = MIMEMultipart("alternative")
    root.attach(MIMEText("<p>HTML</p>", "html"))
    root.attach(MIMEText("PLAIN LINE", "plain"))
    out = am.rfc822_plain_body(root.as_bytes())
    assert "PLAIN LINE" in out
    assert "<p>" not in out


def test_strip_accidental_plist_xml_prefix() -> None:
    from brain.ingest import apple_mail as am

    blob = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<plist version="1.0"><dict><key>k</key><integer>1</integer></dict></plist>'
        "\n\nActual email lines follow.\n"
    )
    assert am._strip_accidental_plist_xml_prefix(blob).strip().startswith("Actual email")


def test_format_envelope_date_apple_epoch() -> None:
    from brain.ingest import apple_mail as am

    # ~2024 via CFAbsoluteTime seconds since 2001 (order of 7e8)
    apple_sec = 725_000_000.0
    s = am._format_envelope_date(apple_sec)
    assert "UTC" in s
    assert "1970" not in s
