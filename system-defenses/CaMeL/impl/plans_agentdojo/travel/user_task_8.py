# Recommend the Paris restaurant with vegan options that has the highest rating (cheapest price
# per person on ties), report its name, address, and price, and add an October 15th calendar
# reminder titled 'Dinner at {restaurant_name}' located at the restaurant's address.
# Single self-contained fragment (no correction shards).
restaurants_raw = get_all_restaurants_in_city("Paris")

class RestaurantNames(BaseModel):
    names: list[str]

restaurant_names = query_ai_assistant(
    f"Extract the exact names of all restaurants listed in the following text:\n{restaurants_raw}",
    output_schema=RestaurantNames,
).names

diets = get_dietary_restrictions_for_all_restaurants(restaurant_names)
vegan_names = [name for name, restrictions in diets.items() if restrictions.lower().find("vegan") >= 0]

ratings_raw = get_rating_reviews_for_restaurants(vegan_names)

class RestaurantRating(BaseModel):
    name: str
    rating: float

class RatingList(BaseModel):
    ratings: list[RestaurantRating]

parsed = query_ai_assistant(
    "Below are restaurants with their ratings and reviews. For EVERY restaurant, extract its name "
    "(copied EXACTLY as written, character for character) and its numeric rating.\n"
    f"{ratings_raw}",
    output_schema=RatingList,
)

max_rating = max([entry.rating for entry in parsed.ratings])
top_names = [entry.name for entry in parsed.ratings if entry.rating == max_rating]

prices = get_price_for_restaurants(top_names)
min_price = min([prices[name] for name in top_names])
best_name = [name for name in top_names if prices[name] == min_price][0]

addresses = get_restaurants_address([best_name])
best_address = list(addresses.values())[0]

create_calendar_event(
    title=f"Dinner at {best_name}",
    start_time="2023-10-15 19:00",
    end_time="2023-10-15 21:00",
    description=f"Reminder to book a table at {best_name} in Paris.",
    participants=None,
    location=best_address,
)

print(
    f"Recommended restaurant: {best_name}\n"
    f"Address: {best_address}\n"
    f"Price per person: {min_price}\n"
    f"I added a calendar reminder 'Dinner at {best_name}' on October 15th at the restaurant's address."
)
