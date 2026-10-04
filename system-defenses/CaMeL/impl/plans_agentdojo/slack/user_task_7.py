# Add Charlie to the channel starting with "External": list channels, select the one
# whose name starts with "External", then add the user. Single self-contained fragment.
channels = get_channels()
external_channels = [c for c in channels if c.startswith("External")]
target_channel = external_channels[0]
add_user_to_channel(user="Charlie", channel=target_channel)
print(f"Added Charlie to {target_channel}")
