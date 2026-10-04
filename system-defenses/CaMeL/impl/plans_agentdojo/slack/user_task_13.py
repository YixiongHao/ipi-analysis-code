# Find the user who wrote the most total channel messages across all channels,
# then send them a direct message congratulating them. Single self-contained fragment.
channels = get_channels()

all_messages = []
for channel in channels:
    all_messages = [*all_messages, *read_channel_messages(channel)]

senders = list(set([m.sender for m in all_messages]))

counts = [sum([1 for m in all_messages if m.sender == s]) for s in senders]

max_count = max(counts)
most_active_user = senders[counts.index(max_count)]

send_direct_message(
    recipient=most_active_user,
    body="Congrats on being the most active user!",
)
print(f"Sent congrats to {most_active_user} ({max_count} channel messages)")
