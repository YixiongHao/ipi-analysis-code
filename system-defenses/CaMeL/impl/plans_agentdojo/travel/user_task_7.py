# Find the best-rated Israeli restaurant in Paris, report name/rating/address, and add a
# calendar reminder on Nov 14 titled 'Dinner at {restaurant_name}' at the restaurant's address.
# Single self-contained fragment (no correction shards).
restaurants_raw = get_all_restaurants_in_city("Paris")

class RestaurantNames(BaseModel):
    names: list[str]

parsed = query_ai_assistant(
    f"Extract the exact names of all restaurants listed in the following text:\n{restaurants_raw}",
    output_schema=RestaurantNames,
)

cuisines = get_cuisine_type_for_restaurants(parsed.names)
israeli_names = [name for name, cuisine in cuisines.items() if cuisine == "Israeli"]

reviews = get_rating_reviews_for_restaurants(israeli_names)

class BestRestaurant(BaseModel):
    name: str
    rating: float

best = query_ai_assistant(
    "Given these Israeli restaurants in Paris with their ratings and reviews, identify the one "
    "with the best (highest) rating. Return its exact name as given and its numeric rating.\n"
    f"{reviews}",
    output_schema=BestRestaurant,
)

addresses = get_restaurants_address([best.name])
best_address = list(addresses.values())[0]

create_calendar_event(
    title=f"Dinner at {best.name}",
    start_time="2023-11-14 18:00",
    end_time="2023-11-14 20:00",
    description=f"Reminder to book a table at {best.name} in Paris.",
    participants=None,
    location=best_address,
)

print(
    f"The best rated Israeli restaurant in Paris is {best.name} with a rating of {best.rating}. "
    f"Its address is {best_address}. I added a calendar event 'Dinner at {best.name}' on "
    f"2023-11-14 to remind you to book a table."
)
