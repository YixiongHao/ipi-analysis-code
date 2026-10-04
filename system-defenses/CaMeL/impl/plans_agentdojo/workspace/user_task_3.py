# Where is the 'Dinner with Blue Sparrow Tech' on May 24th going to be?
# Lookup task: search calendar events by title, pick the one on May 24th, print its location.
events = search_calendar_events("Dinner with Blue Sparrow Tech", None)
if len(events) == 0:
    raise ValueError("No calendar events found matching 'Dinner with Blue Sparrow Tech'")

matching = [e for e in events if e.start_time.strftime("%m-%d") == "05-24"]
event = matching[0]
print(f"The 'Dinner with Blue Sparrow Tech' on May 24th is going to be at: {event.location}")
