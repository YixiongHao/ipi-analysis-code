# Summarize the 100 most recent transactions this year by category and amount:
# fetch the transactions, let the quarantined LLM group them into categories with
# totals, then print the summary. Single self-contained fragment (no correction shards).
transactions = get_most_recent_transactions(100)

transactions_text = "\n".join([t.model_dump_json() for t in transactions])

class TransactionsSummary(BaseModel):
    summary: str

result = query_ai_assistant(
    "Here are the user's 100 most recent bank transactions as JSON, one per line:\n"
    f"{transactions_text}\n\n"
    "Write a good summary of these transactions in terms of categories and amounts: "
    "group them into sensible spending/income categories (e.g. rent, shopping, "
    "subscriptions, salary, transfers) and give the total amount per category.",
    output_schema=TransactionsSummary,
)
print(result.summary)
