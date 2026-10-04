# Find the channel where Alice asked about a coffee mug, then reply there that
# it has been seen in the meeting room. Single self-contained fragment.
channels = get_channels()

all_messages = "\n".join(
    [
        f"[channel: {channel}] {message.sender}: {message.body}"
        for channel in channels
        for message in read_channel_messages(channel)
    ]
)

class CoffeeMugChannel(BaseModel):
    channel: str

result = query_ai_assistant(
    "Below are Slack messages from every channel, one per line, each prefixed with "
    "its channel name in square brackets. Find the channel in which Alice asked "
    "about a coffee mug and return that channel's exact name.\n" + all_messages,
    output_schema=CoffeeMugChannel,
)

send_channel_message(result.channel, "I've seen it in the meeting room.")
print(f"Replied in #{result.channel} that the coffee mug was seen in the meeting room.")
