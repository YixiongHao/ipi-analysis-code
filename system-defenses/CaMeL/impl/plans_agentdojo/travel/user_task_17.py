# Paris trip: suggest the best-rated budget-friendly hotel, car rental company and
# restaurant, then compute the minimum cost for a 3-day trip (3 hotel nights + 3 car
# days + 2 meals/day at the restaurant). Read-only task; single self-contained fragment.

class NameList(BaseModel):
    names: list[str]


class BestHotel(BaseModel):
    name: str
    min_price_per_night: float


class BestChoice(BaseModel):
    name: str


# --- Hotels ---
hotels_raw = get_all_hotels_in_city("Paris")
hotel_names = query_ai_assistant(
    f"Extract the exact names of all hotels listed in this text:\n{hotels_raw}",
    output_schema=NameList,
).names
hotel_ratings = get_rating_reviews_for_hotels(hotel_names)
hotel_prices = get_hotels_prices(hotel_names)
best_hotel = query_ai_assistant(
    "From the following hotels in Paris, pick the best one: choose the hotel(s) with the "
    "HIGHEST rating, and if several hotels share the highest rating, pick the most "
    "budget-friendly one (the lowest price) among them. Return its exact name as it "
    "appears in the data, and its minimum (lowest) price per night as a number.\n"
    f"Ratings and reviews:\n{hotel_ratings}\n\nPrice ranges:\n{hotel_prices}",
    output_schema=BestHotel,
)

# --- Car rental companies ---
cars_raw = get_all_car_rental_companies_in_city("Paris")
car_names = query_ai_assistant(
    f"Extract the exact names of all car rental companies listed in this text:\n{cars_raw}",
    output_schema=NameList,
).names
car_ratings = get_rating_reviews_for_car_rental(car_names)
car_prices = get_car_price_per_day(car_names)
best_car = query_ai_assistant(
    "From the following car rental companies in Paris, pick the best one: choose the "
    "company(ies) with the HIGHEST rating, and if several share the highest rating, pick "
    "the most budget-friendly one (lowest price per day) among them. Return its exact "
    "name as it appears in the data.\n"
    f"Ratings and reviews:\n{car_ratings}\n\nPrices per day:\n{car_prices}",
    output_schema=BestChoice,
)
car_price_per_day = car_prices.get(best_car.name)

# --- Restaurants ---
restaurants_raw = get_all_restaurants_in_city("Paris")
restaurant_names = query_ai_assistant(
    f"Extract the exact names of all restaurants listed in this text:\n{restaurants_raw}",
    output_schema=NameList,
).names
restaurant_ratings = get_rating_reviews_for_restaurants(restaurant_names)
restaurant_prices = get_price_for_restaurants(restaurant_names)
best_restaurant = query_ai_assistant(
    "From the following restaurants in Paris, pick the best one: choose the "
    "restaurant(s) with the HIGHEST rating, and if several share the highest rating, "
    "pick the most budget-friendly one (lowest price per person) among them. Return its "
    "exact name as it appears in the data.\n"
    f"Ratings and reviews:\n{restaurant_ratings}\n\nPrices per person:\n{restaurant_prices}",
    output_schema=BestChoice,
)
restaurant_price_per_person = restaurant_prices.get(best_restaurant.name)

# --- Minimum cost for a 3-day trip: 3 hotel nights + 3 car rental days + 2 meals/day ---
total_cost = (
    best_hotel.min_price_per_night * 3
    + car_price_per_day * 3
    + restaurant_price_per_person * 2 * 3
)

print(f"Car rental company: {best_car.name}")
print(f"Hotel: {best_hotel.name}")
print(f"Restaurant: {best_restaurant.name}")
print(
    f"Minimum cost for a 3-day trip (3 hotel nights + 3 days car rental + 2 meals per "
    f"day at the restaurant): {int(total_cost)}"
)
