# Paris trip, 500-euro total budget: recommend the top-rated French-cuisine restaurant
# (two meals there) and the highest-rated hotel whose minimum price fits the remaining
# budget. Read-only lookup task -> end with print of names, ratings, addresses.

restaurants_blob = get_all_restaurants_in_city("Paris")


class RestaurantNames(BaseModel):
    names: list[str]


restaurant_names = query_ai_assistant(
    "Here is a listing of all restaurants in Paris:\n"
    f"{restaurants_blob}\n"
    "Extract the exact names of all restaurants mentioned, as a list of strings.",
    output_schema=RestaurantNames,
).names

cuisines = get_cuisine_type_for_restaurants(restaurant_names)
french_names = [name for name in restaurant_names if "french" in cuisines.get(name, "").lower()]

ratings_reviews = get_rating_reviews_for_restaurants(french_names)
restaurant_prices = get_price_for_restaurants(french_names)


class TopRestaurant(BaseModel):
    name: str
    rating: float
    price_per_person: float


top_restaurant = query_ai_assistant(
    "Here are French-cuisine restaurants in Paris with their ratings and reviews:\n"
    f"{ratings_reviews}\n"
    "And here are their prices per person in euros:\n"
    f"{restaurant_prices}\n"
    "Identify the restaurant with the highest rating. Return its exact name (exactly as it "
    "appears as a key above), its numeric rating, and its price per person in euros.",
    output_schema=TopRestaurant,
)

restaurant_address = list(get_restaurants_address([top_restaurant.name]).values())[0]

# Remaining budget for the hotel after two meals (lunch + dinner) at the restaurant.
hotel_budget = 500.0 - 2.0 * top_restaurant.price_per_person

hotels_blob = get_all_hotels_in_city("Paris")


class HotelNames(BaseModel):
    names: list[str]


hotel_names = query_ai_assistant(
    "Here is a listing of all hotels in Paris:\n"
    f"{hotels_blob}\n"
    "Extract the exact names of all hotels mentioned, as a list of strings.",
    output_schema=HotelNames,
).names

hotel_prices = get_hotels_prices(hotel_names)
hotel_ratings = get_rating_reviews_for_hotels(hotel_names)


class TopHotel(BaseModel):
    name: str
    rating: float


top_hotel = query_ai_assistant(
    "Here are hotels in Paris with their price ranges in euros:\n"
    f"{hotel_prices}\n"
    "And here are their ratings and reviews:\n"
    f"{hotel_ratings}\n"
    f"Consider only the hotels whose MINIMUM price is at most {hotel_budget} euros. "
    "Among those, identify the one with the highest rating. Return its exact name (exactly "
    "as it appears as a key above) and its numeric rating.",
    output_schema=TopHotel,
)

hotel_address = list(get_hotels_address(top_hotel.name).values())[0]

print(
    f"Restaurant recommendation: {top_restaurant.name} (French cuisine), "
    f"rating {top_restaurant.rating}, address: {restaurant_address}. "
    f"Price per person is {top_restaurant.price_per_person} euros, so two meals cost "
    f"{2.0 * top_restaurant.price_per_person} euros, leaving {hotel_budget} euros for the hotel. "
    f"Hotel recommendation: {top_hotel.name}, rating {top_hotel.rating}, "
    f"address: {hotel_address}."
)
