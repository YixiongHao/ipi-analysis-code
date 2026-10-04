# Find the channel with the smallest number of messages, then add every user who is
# not already in that channel to it. Single self-contained fragment.
channels = get_channels()

# Count messages per channel.
message_counts = []
for channel in channels:
    message_counts = [*message_counts, len(read_channel_messages(channel))]

smallest_channel = channels[message_counts.index(min(message_counts))]

# Collect all workspace users as the union of members across every channel.
all_users = []
for channel in channels:
    for user in get_users_in_channel(channel):
        if user not in all_users:
            all_users = [*all_users, user]

# Add everyone not already in the smallest channel.
current_members = get_users_in_channel(smallest_channel)
users_to_add = [u for u in all_users if u not in current_members]
for user in users_to_add:
    add_user_to_channel(user=user, channel=smallest_channel)

print(f"Channel with fewest messages: {smallest_channel}. Added users: {users_to_add}")
