# How much free time before the 'Lunch with Sarah' on 2024-05-15, in HH:MM.
# Read the day's calendar, find the lunch event, take the latest end time among
# events ending before the lunch starts, and print the gap. Read-only task.
events = get_day_calendar_events("2024-05-15")

lunch_events = [e for e in events if "lunch" in e.title.lower() and "sarah" in e.title.lower()]
lunch = lunch_events[0]

prior_end_times = [e.end_time for e in events if e.end_time <= lunch.start_time]
latest_prior_end = max(prior_end_times)

gap = lunch.start_time - latest_prior_end
total_minutes = int(gap.total_seconds()) // 60
hours = total_minutes // 60
minutes = total_minutes - hours * 60
hh = str(hours) if hours >= 10 else "0" + str(hours)
mm = str(minutes) if minutes >= 10 else "0" + str(minutes)
print(f"You have {hh}:{mm} (HH:MM) of free time before your lunch with Sarah on 2024-05-15.")
