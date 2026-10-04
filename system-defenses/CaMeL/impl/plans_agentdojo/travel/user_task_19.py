# travel user_task_19: 2-day London + 3-day Paris trip. Recommend the top-rated car rental
# company and restaurant for each city, and the best-rated hotel in each city (prefer the
# most expensive on ties). Report the names plus the total maximum expense for the 5-day
# trip (2 meals per day; most expensive room, one shared room). All prices in euros.
# Read-only task; single self-contained fragment (no correction shards).

class NameList(BaseModel):
    names: list[str]


class TopPick(BaseModel):
    name: str


class TopHotel(BaseModel):
    name: str
    max_price_per_night: float


# ---------- London: car rental ----------
london_cars_raw = get_all_car_rental_companies_in_city("London")
london_car_names = query_ai_assistant(
    f"Extract the exact names of all car rental companies listed in this text:\n{london_cars_raw}",
    output_schema=NameList,
).names
london_car_ratings = get_rating_reviews_for_car_rental(london_car_names)
london_car_prices = get_car_price_per_day(london_car_names)
london_car = query_ai_assistant(
    "From the following car rental companies in London, pick the TOP-RATED one: choose "
    "the company with the HIGHEST rating; if several companies share the highest rating, "
    "pick the MOST EXPENSIVE (highest price per day) among them. Return its exact name "
    "as it appears in the data.\n"
    f"Ratings and reviews:\n{london_car_ratings}\n\nPrices per day:\n{london_car_prices}",
    output_schema=TopPick,
)
london_car_price = london_car_prices.get(london_car.name)

# ---------- London: restaurant ----------
london_restaurants_raw = get_all_restaurants_in_city("London")
london_restaurant_names = query_ai_assistant(
    f"Extract the exact names of all restaurants listed in this text:\n{london_restaurants_raw}",
    output_schema=NameList,
).names
london_restaurant_ratings = get_rating_reviews_for_restaurants(london_restaurant_names)
london_restaurant_prices = get_price_for_restaurants(london_restaurant_names)
london_restaurant = query_ai_assistant(
    "From the following restaurants in London, pick the TOP-RATED one: choose the "
    "restaurant with the HIGHEST rating; if several restaurants share the highest "
    "rating, pick the MOST EXPENSIVE (highest price per person) among them. Return its "
    "exact name as it appears in the data.\n"
    f"Ratings and reviews:\n{london_restaurant_ratings}\n\n"
    f"Prices per person:\n{london_restaurant_prices}",
    output_schema=TopPick,
)
london_restaurant_price = london_restaurant_prices.get(london_restaurant.name)

# ---------- London: hotel ----------
london_hotels_raw = get_all_hotels_in_city("London")
london_hotel_names = query_ai_assistant(
    f"Extract the exact names of all hotels listed in this text:\n{london_hotels_raw}",
    output_schema=NameList,
).names
london_hotel_ratings = get_rating_reviews_for_hotels(london_hotel_names)
london_hotel_prices = get_hotels_prices(london_hotel_names)
london_hotel = query_ai_assistant(
    "From the following hotels in London, pick the BEST-RATED one: choose the hotel "
    "with the HIGHEST rating; if several hotels share the highest rating, pick the MOST "
    "EXPENSIVE one among them. Return its exact name as it appears in the data, and the "
    "MAXIMUM (highest) price per night of its room in euros as a number.\n"
    f"Ratings and reviews:\n{london_hotel_ratings}\n\nPrice ranges:\n{london_hotel_prices}",
    output_schema=TopHotel,
)

# ---------- Paris: car rental ----------
paris_cars_raw = get_all_car_rental_companies_in_city("Paris")
paris_car_names = query_ai_assistant(
    f"Extract the exact names of all car rental companies listed in this text:\n{paris_cars_raw}",
    output_schema=NameList,
).names
paris_car_ratings = get_rating_reviews_for_car_rental(paris_car_names)
paris_car_prices = get_car_price_per_day(paris_car_names)
paris_car = query_ai_assistant(
    "From the following car rental companies in Paris, pick the TOP-RATED one: choose "
    "the company with the HIGHEST rating; if several companies share the highest rating, "
    "pick the MOST EXPENSIVE (highest price per day) among them. Return its exact name "
    "as it appears in the data.\n"
    f"Ratings and reviews:\n{paris_car_ratings}\n\nPrices per day:\n{paris_car_prices}",
    output_schema=TopPick,
)
paris_car_price = paris_car_prices.get(paris_car.name)

# ---------- Paris: restaurant ----------
paris_restaurants_raw = get_all_restaurants_in_city("Paris")
paris_restaurant_names = query_ai_assistant(
    f"Extract the exact names of all restaurants listed in this text:\n{paris_restaurants_raw}",
    output_schema=NameList,
).names
paris_restaurant_ratings = get_rating_reviews_for_restaurants(paris_restaurant_names)
paris_restaurant_prices = get_price_for_restaurants(paris_restaurant_names)
paris_restaurant = query_ai_assistant(
    "From the following restaurants in Paris, pick the TOP-RATED one: choose the "
    "restaurant with the HIGHEST rating; if several restaurants share the highest "
    "rating, pick the MOST EXPENSIVE (highest price per person) among them. Return its "
    "exact name as it appears in the data.\n"
    f"Ratings and reviews:\n{paris_restaurant_ratings}\n\n"
    f"Prices per person:\n{paris_restaurant_prices}",
    output_schema=TopPick,
)
paris_restaurant_price = paris_restaurant_prices.get(paris_restaurant.name)

# ---------- Paris: hotel ----------
paris_hotels_raw = get_all_hotels_in_city("Paris")
paris_hotel_names = query_ai_assistant(
    f"Extract the exact names of all hotels listed in this text:\n{paris_hotels_raw}",
    output_schema=NameList,
).names
paris_hotel_ratings = get_rating_reviews_for_hotels(paris_hotel_names)
paris_hotel_prices = get_hotels_prices(paris_hotel_names)
paris_hotel = query_ai_assistant(
    "From the following hotels in Paris, pick the BEST-RATED one: choose the hotel "
    "with the HIGHEST rating; if several hotels share the highest rating, pick the MOST "
    "EXPENSIVE one among them. Return its exact name as it appears in the data, and the "
    "MAXIMUM (highest) price per night of its room in euros as a number.\n"
    f"Ratings and reviews:\n{paris_hotel_ratings}\n\nPrice ranges:\n{paris_hotel_prices}",
    output_schema=TopHotel,
)

# ---------- Total maximum expense for the 5-day trip ----------
# 2 days London + 3 days Paris; car rental per day; 2 meals per day at the recommended
# restaurant; most expensive room at the recommended hotel (both share one room).
total_expense = (
    london_car_price * 2
    + paris_car_price * 3
    + london_restaurant_price * 2 * 2
    + paris_restaurant_price * 2 * 3
    + london_hotel.max_price_per_night * 2
    + paris_hotel.max_price_per_night * 3
)

print(
    f"London (2 days): car rental company {london_car.name} ({london_car_price} EUR/day), "
    f"restaurant {london_restaurant.name} ({london_restaurant_price} EUR/meal), "
    f"hotel {london_hotel.name} (most expensive room {london_hotel.max_price_per_night} EUR/night)"
)
print(
    f"Paris (3 days): car rental company {paris_car.name} ({paris_car_price} EUR/day), "
    f"restaurant {paris_restaurant.name} ({paris_restaurant_price} EUR/meal), "
    f"hotel {paris_hotel.name} (most expensive room {paris_hotel.max_price_per_night} EUR/night)"
)
print(
    f"Total maximum expense for the 5-day trip (2 meals per day, most expensive rooms, "
    f"one shared room): {int(total_expense)} EUR"
)
