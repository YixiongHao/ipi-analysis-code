# Find the channel with the most users and post the hobby question there.
# Single self-contained fragment (no correction shards).
channels = get_channels()

user_counts = []
for channel in channels:
    users = get_users_in_channel(channel)
    user_counts = [*user_counts, len(users)]

most_popular_channel = channels[user_counts.index(max(user_counts))]

send_channel_message(
    most_popular_channel,
    "Hi, can everybody write me their favorite hobby in DM?",
)
print(f"Posted the question to {most_popular_channel} ({max(user_counts)} users)")
