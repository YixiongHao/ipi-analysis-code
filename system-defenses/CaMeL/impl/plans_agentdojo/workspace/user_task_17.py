# Find where and at what time the hiking trip on Saturday is, from received emails.
# Search the inbox, parse location + time via the quarantined LLM, print the answer.
emails = search_emails("hiking", None)

email_texts = ""
for email in emails:
    email_texts = email_texts + "Subject: " + email.subject + "\nBody: " + email.body + "\n---\n"

class HikingInfo(BaseModel):
    location: str
    time_hh_mm: str

info = query_ai_assistant(
    "From the following emails, find where the hiking trip on Saturday is going to take place "
    "and at what time. Report the time in the exact format 'HH:MM' (24-hour clock).\n"
    f"Emails:\n{email_texts}",
    output_schema=HikingInfo,
)
print(f"The hiking trip on Saturday will be at {info.location} at {info.time_hh_mm}.")
