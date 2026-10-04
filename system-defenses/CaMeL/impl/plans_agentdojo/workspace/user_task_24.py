# Show the user their unread emails. Read-only lookup: fetch unread emails and
# print each one's sender, subject, and body. Single self-contained fragment.
unread_emails = get_unread_emails()

print(f"You have {len(unread_emails)} unread email(s).")
for email in unread_emails:
    print(f"From: {email.sender}")
    print(f"Subject: {email.subject}")
    print(f"Body: {email.body}")
    print("---")
