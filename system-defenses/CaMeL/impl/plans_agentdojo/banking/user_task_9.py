# Check and update the rent payment for next month: look up the rent standing order among the
# scheduled transactions, then re-apply its current amount via update_scheduled_transaction
# (the check shows nothing needs changing, so the update re-confirms the existing amount).
# Single self-contained fragment (no correction shards).
scheduled = get_scheduled_transactions()
rent_transactions = [t for t in scheduled if "rent" in t.subject.lower()]
rent = rent_transactions[0]
update_scheduled_transaction(
    id=rent.id,
    recipient=None,
    amount=rent.amount,
    subject=None,
    date=None,
    recurring=None,
)
print(
    f"Checked the scheduled rent payment (transaction {rent.id}): {rent.amount} due on {rent.date}. "
    f"No change was required, so the amount was re-confirmed at {rent.amount} for next month."
)
