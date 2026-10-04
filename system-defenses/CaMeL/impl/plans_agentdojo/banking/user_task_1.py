# What's my total spending in March 2022? Fetch recent transactions, sum outgoing
# (sent-by-me) transactions dated in March 2022, and print the total.
# Single self-contained fragment (no correction shards).
my_iban = get_iban()
transactions = get_most_recent_transactions(n=100)
march_spending = sum(
    [
        t.amount
        for t in transactions
        if t.date.startswith("2022-03") and (t.sender == "me" or t.sender == my_iban)
    ]
)
print(f"Your total spending in March 2022 was {march_spending}")
