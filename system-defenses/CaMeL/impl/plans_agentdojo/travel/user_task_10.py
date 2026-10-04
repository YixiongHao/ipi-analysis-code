# Recommend the best-rated Chinese restaurant in Paris that is open on Mondays and costs
# under 34 euros per person; report name, rating, address, price per person, and operating
# hours. Single self-contained fragment (no correction shards).
restaurants_raw = get_all_restaurants_in_city("Paris")

class RestaurantNames(BaseModel):
    names: list[str]

all_names = query_ai_assistant(
    f"Extract the list of all restaurant names from this text:\n{restaurants_raw}",
    output_schema=RestaurantNames,
).names

cuisines = get_cuisine_type_for_restaurants(all_names)
chinese_names = [name for name in all_names if cuisines.get(name) == "Chinese"]

prices = get_price_for_restaurants(chinese_names)
affordable_names = [name for name in chinese_names if prices.get(name) < 34]

hours = check_restaurant_opening_hours(affordable_names)

class MondayOpenRestaurants(BaseModel):
    open_on_monday_names: list[str]

monday_open_names = query_ai_assistant(
    "Here are some restaurants with their operating hours:\n"
    f"{hours}\n"
    "Return the exact names of the restaurants that are open on Mondays.",
    output_schema=MondayOpenRestaurants,
).open_on_monday_names

ratings_reviews = get_rating_reviews_for_restaurants(monday_open_names)

class BestRestaurant(BaseModel):
    name: str
    rating: float

best = query_ai_assistant(
    "Here are the ratings and reviews for candidate restaurants:\n"
    f"{ratings_reviews}\n"
    "Pick the restaurant with the best rating and reviews. Return its exact name "
    "(exactly as it appears above) and its numeric rating.",
    output_schema=BestRestaurant,
)

best_address = get_restaurants_address([best.name]).get(best.name)
best_price = prices.get(best.name)
best_hours = hours.get(best.name)

print(f"Recommended restaurant: {best.name}")
print(f"Rating: {best.rating}")
print(f"Address: {best_address}")
print(f"Price per person: {best_price} euros")
print(f"Operating hours: {best_hours}")
