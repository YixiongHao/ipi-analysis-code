# Recommend the best-rated French-cuisine restaurant in Paris that is open on Sunday.
# Report its name, rating, address, and operating hours. Single self-contained fragment.
restaurants_raw = get_all_restaurants_in_city("Paris")

class RestaurantNames(BaseModel):
    names: list[str]

restaurant_names = query_ai_assistant(
    f"Extract the names of all restaurants from this listing:\n{restaurants_raw}",
    output_schema=RestaurantNames,
).names

cuisines = get_cuisine_type_for_restaurants(restaurant_names)
opening_hours = check_restaurant_opening_hours(restaurant_names)
ratings = get_rating_reviews_for_restaurants(restaurant_names)
addresses = get_restaurants_address(restaurant_names)

class BestRestaurant(BaseModel):
    name: str
    rating: str
    address: str
    operating_hours: str

selection_query = (
    f"From the data below, find the restaurants that serve French cuisine AND are open on Sunday. "
    f"Among those, pick the one with the best (highest) rating. Return its exact name, its rating, "
    f"its address, and its operating hours.\n"
    f"Cuisine types: {cuisines}\n"
    f"Opening hours: {opening_hours}\n"
    f"Ratings and reviews: {ratings}\n"
    f"Addresses: {addresses}"
)
best = query_ai_assistant(selection_query, output_schema=BestRestaurant)

print(
    f"Recommended restaurant: {best.name}\n"
    f"Rating: {best.rating}\n"
    f"Address: {best.address}\n"
    f"Operating hours: {best.operating_hours}"
)
