"""Due dates. Payment is due `net_days` after issue; if that falls on a weekend,
it moves to the following Monday."""

from datetime import timedelta


def due_date(issued, net_days):
    due = issued + timedelta(days=net_days)
    if due.weekday() == 5:  # Saturday
        due += timedelta(days=2)
    return due
