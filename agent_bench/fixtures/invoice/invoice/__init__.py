"""A tiny invoicing library used as a fixed agent test. See ../README.md."""

from .dates import due_date
from .discount import apply_discount, discount_rate
from .money import format_money, round_money

TAX_RATE = 0.13


def invoice_total(items, issued, net_days=30):
    """items: list of (unit_price, quantity). Discount first, then tax, then round."""
    subtotal = sum(price * qty for price, qty in items)
    discounted = apply_discount(subtotal)
    total = round_money(discounted * (1 + TAX_RATE))
    return {
        "subtotal": round_money(subtotal),
        "discount_rate": discount_rate(subtotal),
        "total": total,
        "total_text": format_money(total),
        "due": due_date(issued, net_days),
    }
