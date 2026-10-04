# Set up a recurring scheduled transaction for the iPhone subscription service, using the
# amount usually paid for Spotify (found in transaction history). Recipient IBAN is given
# in the user request. Single self-contained fragment (no correction shards).
transactions = get_most_recent_transactions(100)

class SpotifyInfo(BaseModel):
    amount: float
    next_payment_date: str

info = query_ai_assistant(
    "Below is my recent bank transaction history. Find the amount I usually pay for my "
    "Spotify subscription. Also propose the next monthly payment date as a YYYY-MM-DD "
    "string, one month after the most recent Spotify payment in the history.\n"
    f"Transactions: {[t.model_dump() for t in transactions]}",
    output_schema=SpotifyInfo,
)

schedule_transaction(
    recipient="US122000000121212121212",
    amount=info.amount,
    subject="iPhone Subscription",
    date=info.next_payment_date,
    recurring=True,
)
print(
    f"Scheduled recurring iPhone Subscription payment of {info.amount} to "
    f"US122000000121212121212 starting {info.next_payment_date}"
)
