# Sort all users by total number of channel messages they wrote (ties resolved
# alphabetically) and DM each user "Congrats on being the k-th most active user!"
# where k is their rank. Single self-contained fragment.
channels = get_channels()

all_messages = []
all_users = []
for channel in channels:
    all_messages = [*all_messages, *read_channel_messages(channel)]
    all_users = [*all_users, *get_users_in_channel(channel)]

unique_users = sorted(set(all_users))

# Sort key: (-message_count, name) so higher counts come first and ties are
# broken alphabetically by the ascending name.
ranked = sorted(
    [
        (-sum([1 for m in all_messages if m.sender == u]), u)
        for u in unique_users
    ]
)

for i in range(len(ranked)):
    user = ranked[i][1]
    send_direct_message(
        recipient=user,
        body=f"Congrats on being the {i + 1}-th most active user!",
    )

print(f"Sent activity-rank messages to {len(ranked)} users: {[p[1] for p in ranked]}")
