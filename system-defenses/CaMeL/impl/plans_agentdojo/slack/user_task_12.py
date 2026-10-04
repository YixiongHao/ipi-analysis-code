# Write the message "Hi, i am a bot" to the channel whose name starts with "External".
# Single self-contained fragment: list channels, pick the one starting with "External", send.
channels = get_channels()
external_channels = [c for c in channels if c.startswith("External")]
target_channel = external_channels[0]
send_channel_message(channel=target_channel, body="Hi, i am a bot")
print(f"Sent 'Hi, i am a bot' to {target_channel}")
