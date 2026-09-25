"""Money helpers. All amounts are in yuan with two decimal places."""


def round_money(amount):
    """Round to cents, half away from zero (0.125 -> 0.13, 2.675 -> 2.68).

    This is the rounding rule printed on every invoice, so it must not depend on
    how a value happens to be stored in binary floating point.
    """
    return round(amount, 2)


def format_money(amount):
    """Format as a yuan string with thousands separators: 1234.5 -> '¥1,234.50'."""
    return "¥{:,.2f}".format(round_money(amount))
