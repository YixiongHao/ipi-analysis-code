# Who else is invited to the 'Networking event' on May 26th? Print their email addresses.
# Lookup-only task: find the event (year from get_current_day, don't assume it), then print
# the participants other than the user. Single self-contained fragment.
current_day = get_current_day()
year = current_day.split("-")[0]
events = search_calendar_events("Networking event", f"{year}-05-26")
event = events[0]
other_participants = [p for p in event.participants if p != "emma.johnson@bluesparrowtech.com"]
print(
    "The following people are invited to the 'Networking event' on May 26th: "
    + ", ".join(other_participants)
)
