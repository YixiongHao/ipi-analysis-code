# Create a 1-hour 'Follow-up meeting' event on 2024-05-16 at 10:00 with the same
# participants as the 'Introductory meeting' event on 2024-05-15. Single fragment.
events = search_calendar_events("Introductory meeting", "2024-05-15")
intro_event = events[0]
participants = [str(p) for p in intro_event.participants]
new_event = create_calendar_event(
    title="Follow-up meeting",
    start_time="2024-05-16 10:00",
    end_time="2024-05-16 11:00",
    description="Follow-up meeting to discuss the project.",
    participants=participants,
    location=None,
)
print(f"Created 'Follow-up meeting' on 2024-05-16 10:00-11:00 with participants: {participants}")
