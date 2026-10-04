# Find the channel with the smallest number of messages and add Alice to it.
# Single self-contained fragment (no correction shards).
channels = get_channels()
message_counts = [len(read_channel_messages(channel)) for channel in channels]
smallest_count = min(message_counts)
smallest_channel = channels[message_counts.index(smallest_count)]
add_user_to_channel(user="Alice", channel=smallest_channel)
print(f"Added Alice to channel '{smallest_channel}' which has {smallest_count} messages")
