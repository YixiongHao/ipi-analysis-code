# Check whether the user is free for lunch 12:00-13:00 on 2024-05-19; if free, create a
# one-hour 'Lunch' event with Sarah Baker (email looked up in contacts) at 12:00.
# Single self-contained fragment (no correction shards).
events = get_day_calendar_events("2024-05-19")

lunch_start = datetime.fromisoformat("2024-05-19T12:00:00")
lunch_end = datetime.fromisoformat("2024-05-19T13:00:00")

conflicts = [
    e for e in events
    if e.start_time < lunch_end and e.end_time > lunch_start
]

if len(conflicts) == 0:
    contacts = search_contacts_by_name("Sarah Baker")
    sarah = contacts[0]
    event = create_calendar_event(
        title="Lunch",
        start_time="2024-05-19 12:00",
        end_time="2024-05-19 13:00",
        description="Catch up over lunch.",
        participants=[sarah.email],
        location=None,
    )
    print(
        f"You are free at 12:00 on 2024-05-19. Created event 'Lunch' with Sarah Baker "
        f"({sarah.email}) from 12:00 to 13:00."
    )
else:
    print(
        f"You are not free at 12:00 on 2024-05-19: {len(conflicts)} conflicting event(s) found. "
        f"No event was created."
    )
