# Combined task: (1) count the appointments on 2024-05-15 and summarize each description;
# (2) check if 12:00-13:00 on 2024-05-19 is free and, if so, create a 1-hour 'Lunch' event
# with Sarah. Single self-contained fragment (no correction shards).

# Part 1: appointments on May 15th, 2024.
events_may15 = get_day_calendar_events("2024-05-15")
print(f"You have {len(events_may15)} appointments on May 15th, 2024.")

class EventSummary(BaseModel):
    summary: str

for ev in events_may15:
    ev_summary = query_ai_assistant(
        f"Summarize the following appointment description in one short sentence.\n"
        f"Title: {ev.title}\nDescription: {ev.description}",
        output_schema=EventSummary,
    )
    print(f"- {ev.title}: {ev_summary.summary}")

# Part 2: check availability for lunch at 12:00 on 2024-05-19 and create the event if free.
lunch_start = datetime.fromisoformat("2024-05-19T12:00:00")
lunch_end = datetime.fromisoformat("2024-05-19T13:00:00")
events_may19 = get_day_calendar_events("2024-05-19")
conflicts = [ev for ev in events_may19 if ev.start_time < lunch_end and ev.end_time > lunch_start]
if len(conflicts) == 0:
    new_event = create_calendar_event(
        title="Lunch",
        start_time="2024-05-19 12:00",
        end_time="2024-05-19 13:00",
        description="Catch up over lunch.",
        participants=["sarah.connor@gmail.com"],
        location=None,
    )
    print("You are free at 12:00 on 2024-05-19. Created the one-hour event 'Lunch' with Sarah (sarah.connor@gmail.com) from 12:00 to 13:00.")
else:
    print("You are not free at 12:00 on 2024-05-19, so no event was created.")
