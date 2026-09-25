"""Volume discounts.

Tiers, applied to the subtotal before tax:
    subtotal >= 5000  -> 10% off
    subtotal >= 1000  ->  5% off
    otherwise         ->  no discount
"""

TIERS = [
    (5000, 0.10),
    (1000, 0.05),
]


def discount_rate(subtotal):
    for threshold, rate in TIERS:
        if subtotal > threshold:
            return rate
    return 0.0


def apply_discount(subtotal):
    return subtotal * (1 - discount_rate(subtotal))
