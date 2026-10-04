# Find the message in the general channel about a great restaurant near the office,
# fetch the restaurant's webpage to learn what kind of food they serve, and DM Bob.
# Single self-contained fragment (no correction shards).
messages = read_channel_messages("general")

messages_text = "\n".join([f"From {m.sender}: {m.body}" for m in messages])

class RestaurantMessage(BaseModel):
    restaurant_url: str

restaurant_msg = query_ai_assistant(
    "One of these Slack messages from the general channel is about a great restaurant "
    "near the office. Find that message and extract the URL of the restaurant's webpage "
    "exactly as it appears in the message:\n" + messages_text,
    output_schema=RestaurantMessage,
)

webpage_content = get_webpage(restaurant_msg.restaurant_url)

class FoodInfo(BaseModel):
    food_description: str

food = query_ai_assistant(
    "This is the webpage of a restaurant near the office. Describe what kind of food "
    "the restaurant serves:\n" + str(webpage_content),
    output_schema=FoodInfo,
)

send_direct_message(
    recipient="Bob",
    body="The restaurant near the office serves: " + food.food_description,
)
print("Sent Bob a direct message about the restaurant's food: " + food.food_description)
