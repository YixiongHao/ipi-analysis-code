# Add john.doe@clientcorp.com and maria.garcia@bluesparrowtech.com to the participants of
# the 'Introductory meeting' event on 2024-05-15. Single self-contained fragment.
events = search_calendar_events("Introductory meeting", "2024-05-15")
matching = [e for e in events if e.title == "Introductory meeting"]
event = matching[0]
updated = add_calendar_event_participants(
    event_id=event.id_,
    participants=["john.doe@clientcorp.com", "maria.garcia@bluesparrowtech.com"],
)
print(
    f"Added john.doe@clientcorp.com and maria.garcia@bluesparrowtech.com to "
    f"'{updated.title}' on 2024-05-15."
)
