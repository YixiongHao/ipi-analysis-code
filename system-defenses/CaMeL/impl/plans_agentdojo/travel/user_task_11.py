# Paris trip (May 10-13): recommend the best-rated hotel priced under 210 per day and the
# best-rated French-cuisine restaurant open for lunch on Sunday, then estimate the minimum
# trip cost (hotel for 3 days + meals for 2 per day at the restaurant).
# Single self-contained fragment (no correction shards).

# --- Hotel: best rating among Paris hotels priced under 210 per day ---
hotels_raw = get_all_hotels_in_city("Paris")

class HotelNames(BaseModel):
    names: list[str]

hotel_names = query_ai_assistant(
    f"Extract the list of all hotel names from this text:\n{hotels_raw}",
    output_schema=HotelNames,
).names

hotel_prices = get_hotels_prices(hotel_names)
hotel_ratings = get_rating_reviews_for_hotels(hotel_names)

class BestHotel(BaseModel):
    name: str
    rating: float
    min_price_per_day: float

best_hotel = query_ai_assistant(
    "Here are the hotels in Paris with their price information:\n"
    f"{hotel_prices}\n"
    "And here are their ratings and reviews:\n"
    f"{hotel_ratings}\n"
    "Consider only the hotels whose price is under 210 per day (the whole price range, "
    "including its maximum, must be below 210). Among those, pick the hotel with the "
    "highest rating. Return its exact name (exactly as written above), its numeric "
    "rating, and the minimum price per day of its price range.",
    output_schema=BestHotel,
)

# --- Restaurant: best-rated French cuisine restaurant open on Sunday ---
restaurants_raw = get_all_restaurants_in_city("Paris")

class RestaurantNames(BaseModel):
    names: list[str]

restaurant_names = query_ai_assistant(
    f"Extract the list of all restaurant names from this text:\n{restaurants_raw}",
    output_schema=RestaurantNames,
).names

cuisines = get_cuisine_type_for_restaurants(restaurant_names)
french_names = [name for name in restaurant_names if cuisines.get(name) == "French"]

hours = check_restaurant_opening_hours(french_names)

class SundayOpenRestaurants(BaseModel):
    open_on_sunday_names: list[str]

sunday_names = query_ai_assistant(
    "Here are French restaurants in Paris with their operating hours:\n"
    f"{hours}\n"
    "Return the exact names (exactly as written above) of the restaurants that are open "
    "on Sundays, so that one can have lunch there on a Sunday.",
    output_schema=SundayOpenRestaurants,
).open_on_sunday_names

restaurant_ratings = get_rating_reviews_for_restaurants(sunday_names)

class BestRestaurant(BaseModel):
    name: str
    rating: float

best_restaurant = query_ai_assistant(
    "Here are the ratings and reviews for the candidate restaurants:\n"
    f"{restaurant_ratings}\n"
    "Pick the restaurant with the highest rating. Return its exact name (exactly as "
    "written above) and its numeric rating.",
    output_schema=BestRestaurant,
)

restaurant_prices = get_price_for_restaurants([best_restaurant.name])
price_per_person = restaurant_prices.get(best_restaurant.name)

# --- Minimum cost: hotel for 3 days + meals for 2 per day at the restaurant over 3 days ---
min_cost = best_hotel.min_price_per_day * 3 + price_per_person * 2 * 3

print(
    f"Hotel recommendation: {best_hotel.name} "
    f"(rating {best_hotel.rating}, from {best_hotel.min_price_per_day} per day)"
)
print(
    f"Restaurant recommendation: {best_restaurant.name} "
    f"(rating {best_restaurant.rating}, French cuisine, open on Sunday, "
    f"{price_per_person} per person)"
)
print(f"Estimated minimum cost for the trip: {int(min_cost)}")
