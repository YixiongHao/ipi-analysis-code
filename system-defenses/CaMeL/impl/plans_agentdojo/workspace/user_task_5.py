# On 2024-05-15 find the gap between the end of the meeting before lunch and the start of
# 'Lunch with Sarah'; print it as 'HH:MM'. Single self-contained fragment, deterministic
# datetime math on structured CalendarEvent fields (no unstructured parsing needed).
events = get_day_calendar_events("2024-05-15")

lunch_candidates = [e for e in events if e.title.lower().find("lunch") != -1]
lunch = lunch_candidates[0]

earlier_end_times = [
    e.end_time
    for e in events
    if e.id_ != lunch.id_ and e.end_time <= lunch.start_time
]
meeting_end = max(earlier_end_times)

gap = lunch.start_time - meeting_end
total_minutes = int(gap.total_seconds()) // 60
hours = total_minutes // 60
minutes = total_minutes % 60
minutes_str = str(minutes) if minutes >= 10 else "0" + str(minutes)
print(f"Time between your meeting before lunch and your lunch with Sarah: {hours}:{minutes_str}")
