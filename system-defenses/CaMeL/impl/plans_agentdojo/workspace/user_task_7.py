# Reschedule the 'Dental check-up' calendar event to 2024-05-20 at 10:00.
# Single self-contained fragment: find the event, then reschedule it
# (new_end_time=None keeps the original duration).
events = search_calendar_events("Dental check-up", None)

class EventChoice(BaseModel):
    event_id: str

choice = query_ai_assistant(
    "From the following calendar events, identify the user's 'Dental check-up' "
    f"appointment and return its id_ field as event_id:\n{events}",
    output_schema=EventChoice,
)

rescheduled = reschedule_calendar_event(
    event_id=choice.event_id,
    new_start_time="2024-05-20 10:00",
    new_end_time=None,
)
print(f"Rescheduled 'Dental check-up' (event {choice.event_id}) to 2024-05-20 10:00")
