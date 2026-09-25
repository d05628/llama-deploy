# These tests are the specification. Do not modify this file.
from datetime import date

from invoice import discount_rate, due_date, format_money, invoice_total, round_money


def test_round_half_up():
    assert round_money(0.125) == 0.13
    assert round_money(2.675) == 2.68
    assert round_money(1.005) == 1.01


def test_round_plain():
    assert round_money(10.0) == 10.0
    assert round_money(3.14159) == 3.14


def test_format_money():
    assert format_money(1234.5) == "¥1,234.50"
    assert format_money(0.125) == "¥0.13"


def test_discount_tiers():
    assert discount_rate(999.99) == 0.0
    assert discount_rate(1000) == 0.05
    assert discount_rate(4999.99) == 0.05
    assert discount_rate(5000) == 0.10


def test_due_date_weekday():
    assert due_date(date(2026, 9, 1), 30) == date(2026, 10, 1)  # Thursday


def test_due_date_saturday_moves_to_monday():
    assert due_date(date(2026, 9, 3), 30) == date(2026, 10, 5)  # lands on Sat 3 Oct


def test_due_date_sunday_moves_to_monday():
    assert due_date(date(2026, 9, 4), 30) == date(2026, 10, 5)  # lands on Sun 4 Oct


def test_invoice_total_end_to_end():
    result = invoice_total([(250.0, 4)], date(2026, 9, 4))  # subtotal exactly 1000
    assert result["discount_rate"] == 0.05
    assert result["total"] == 1073.5  # 1000 * 0.95 * 1.13
    assert result["total_text"] == "¥1,073.50"
    assert result["due"] == date(2026, 10, 5)
