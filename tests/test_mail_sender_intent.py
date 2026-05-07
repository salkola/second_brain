from __future__ import annotations

from brain.tools.builtin import wants_mail_sender_breakdown_query


def test_wants_mail_sender_breakdown_categorize_email_senders() -> None:
    assert wants_mail_sender_breakdown_query("categorize all email senders")


def test_wants_mail_sender_breakdown_rank_inbox_senders() -> None:
    assert wants_mail_sender_breakdown_query("rank senders in my inbox by volume")


def test_wants_mail_sender_false_for_unrelated() -> None:
    assert not wants_mail_sender_breakdown_query("summarize my notes about Python")


def test_wants_mail_sender_false_sender_without_mail_context() -> None:
    assert not wants_mail_sender_breakdown_query("who is the sender of this package")


def test_wants_mail_sender_who_emailed() -> None:
    assert wants_mail_sender_breakdown_query("who emailed me the most this year?")
