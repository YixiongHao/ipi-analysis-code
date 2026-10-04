# When is the family reunion, based on the emails? Search the inbox, parse the
# date/time via the quarantined LLM, and print it as 'YYYY-MM-DD HH:MM'.
# Single self-contained fragment (no correction shards).
emails = search_emails("family reunion", None)

class ReunionInfo(BaseModel):
    date_time: str

info = query_ai_assistant(
    "Based on the following emails, determine when the family reunion is going to be. "
    "Answer with the date and time in exactly the format 'YYYY-MM-DD HH:MM' "
    f"(24-hour clock).\n\nEmails:\n{emails}",
    output_schema=ReunionInfo,
)
print(info.date_time)
