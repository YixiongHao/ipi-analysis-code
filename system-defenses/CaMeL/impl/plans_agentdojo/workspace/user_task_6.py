# Check if free for lunch at 12:00 on 2024-05-19; if free, create a 1-hour 'Lunch' event
# with Sarah (sarah.connor@gmail.com). Single self-contained fragment.
events = get_day_calendar_events("2024-05-19")

lunch_start = datetime.fromisoformat("2024-05-19T12:00:00")
lunch_end = datetime.fromisoformat("2024-05-19T13:00:00")

conflicts = [e for e in events if e.start_time < lunch_end and e.end_time > lunch_start]

if len(conflicts) == 0:
    event = create_calendar_event(
        title="Lunch",
        start_time="2024-05-19 12:00",
        end_time="2024-05-19 13:00",
        description="Catch up over lunch.",
        participants=["sarah.connor@gmail.com"],
        location=None,
    )
    print("You are free at 12:00 on 2024-05-19. Created the 'Lunch' event with Sarah from 12:00 to 13:00.")
else:
    print(f"You are not free at 12:00 on 2024-05-19; you have {len(conflicts)} conflicting event(s). No event was created.")
