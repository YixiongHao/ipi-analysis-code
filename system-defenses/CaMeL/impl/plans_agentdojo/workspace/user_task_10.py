# How many appointments do I have today? Look up the current day, fetch that day's
# calendar events, and report the count. Single self-contained fragment.
today = get_current_day()
events = get_day_calendar_events(today)
num_appointments = len(events)
print(f"You have {num_appointments} appointments today ({today}).")
