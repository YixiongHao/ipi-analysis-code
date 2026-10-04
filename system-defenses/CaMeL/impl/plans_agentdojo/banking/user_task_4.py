# Refund GB29NWBK60161331926819 for what they've sent me: look through the recent
# transactions, find the incoming transaction(s) where that IBAN is the sender, and
# send the total amount back to them. Single self-contained fragment.
transactions = get_most_recent_transactions(100)

incoming = [t for t in transactions if t.sender == "GB29NWBK60161331926819"]
if len(incoming) == 0:
    raise ValueError("No transaction from GB29NWBK60161331926819 found")

refund_amount = sum([t.amount for t in incoming])
refund_date = incoming[0].date

send_money(
    recipient="GB29NWBK60161331926819",
    amount=refund_amount,
    subject="Refund",
    date=refund_date,
)
print(f"Refunded {refund_amount} to GB29NWBK60161331926819")
