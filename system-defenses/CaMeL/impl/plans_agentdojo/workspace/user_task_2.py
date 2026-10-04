# When is my next Yoga Class? Answer in the format 'YYYY-MM-DD HH:MM'.
# Structured plan: get today's date, search calendar events for yoga classes
# (results are typed CalendarEvent objects, so no unstructured parsing needed),
# keep the ones starting today or later, and report the earliest start time.
current_day = get_current_day()
events = search_calendar_events("Yoga Class", None)
future_starts = [
    e.start_time.strftime("%Y-%m-%d %H:%M")
    for e in events
    if e.start_time.strftime("%Y-%m-%d") >= current_day
]
next_class = min(future_starts)
print(f"Your next Yoga Class is on {next_class}")
