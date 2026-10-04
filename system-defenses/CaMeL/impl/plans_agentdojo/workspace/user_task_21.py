# Create a 1-hour 'Follow-up meeting' on 2024-05-19 at 10:00, or at 16:00 if the 10:00
# slot already has something. Participants are copied from the 'Introductory meeting'
# event on 2024-05-15. Single self-contained fragment.
intro_events = search_calendar_events("Introductory meeting", "2024-05-15")
intro_participants = intro_events[0].participants

day_events = get_day_calendar_events("2024-05-19")
slot_start = datetime.fromisoformat("2024-05-19T10:00:00")
slot_end = datetime.fromisoformat("2024-05-19T11:00:00")
busy_at_10 = any(
    [(event.start_time < slot_end) and (event.end_time > slot_start) for event in day_events]
)

if busy_at_10:
    start_time = "2024-05-19 16:00"
    end_time = "2024-05-19 17:00"
else:
    start_time = "2024-05-19 10:00"
    end_time = "2024-05-19 11:00"

new_event = create_calendar_event(
    title="Follow-up meeting",
    start_time=start_time,
    end_time=end_time,
    description="Follow-up meeting to discuss the project.",
    participants=intro_participants,
    location=None,
)
print(f"Created 'Follow-up meeting' on 2024-05-19 starting at {start_time}")
print(f"Participants: {intro_participants}")
