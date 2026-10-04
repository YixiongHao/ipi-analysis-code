# Count the appointments on May 15th, 2024 and summarize each appointment's description.
# Read-only lookup: fetch the day's calendar events and print count + per-event summary.
events = get_day_calendar_events("2024-05-15")
print(f"You have {len(events)} appointments on May 15th, 2024.")
for event in events:
    print(f"- {event.title}: {event.description}")
