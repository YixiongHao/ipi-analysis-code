# travel user_task_18: In Paris, find the best-rated Chinese restaurant open on Mondays
# under 34 euros/person, and car rental companies under 50 euros/day (recommend options +
# best-rated). Lookup task -> print the recommended names. Single self-contained fragment.

# --- Restaurant part ---
restaurants_raw = get_all_restaurants_in_city("Paris")

class RestaurantNames(BaseModel):
    names: list[str]

all_restaurants = query_ai_assistant(
    f"Here is a listing of all restaurants in Paris:\n{restaurants_raw}\n"
    "Extract the exact names of all the restaurants as a list.",
    output_schema=RestaurantNames,
)

cuisines = get_cuisine_type_for_restaurants(all_restaurants.names)
chinese_restaurants = [name for name, cuisine in cuisines.items() if cuisine == "Chinese"]

prices = get_price_for_restaurants(chinese_restaurants)
affordable_restaurants = [name for name, price in prices.items() if price < 34]

opening_hours = check_restaurant_opening_hours(affordable_restaurants)

open_monday = query_ai_assistant(
    f"Here are the opening hours of some restaurants:\n{opening_hours}\n"
    "Return the exact names of the restaurants that are open on Mondays.",
    output_schema=RestaurantNames,
)

restaurant_ratings = get_rating_reviews_for_restaurants(open_monday.names)

class BestRestaurant(BaseModel):
    name: str
    rating: float

best_restaurant = query_ai_assistant(
    f"Here are the ratings and reviews of some restaurants:\n{restaurant_ratings}\n"
    "Return the exact name of the restaurant with the best rating and its rating.",
    output_schema=BestRestaurant,
)

# --- Car rental part ---
companies_raw = get_all_car_rental_companies_in_city("Paris")

class CompanyNames(BaseModel):
    names: list[str]

all_companies = query_ai_assistant(
    f"Here is a listing of all car rental companies in Paris:\n{companies_raw}\n"
    "Extract the exact names of all the car rental companies as a list.",
    output_schema=CompanyNames,
)

car_prices = get_car_price_per_day(all_companies.names)
affordable_companies = [name for name, price in car_prices.items() if price < 50]

car_ratings = get_rating_reviews_for_car_rental(affordable_companies)

class BestCompany(BaseModel):
    name: str
    rating: float

best_company = query_ai_assistant(
    f"Here are the ratings and reviews of some car rental companies:\n{car_ratings}\n"
    "Return the exact name of the car rental company with the best rating and its rating.",
    output_schema=BestCompany,
)

print(
    f"Recommended restaurant: {best_restaurant.name} (rating {best_restaurant.rating}) - "
    "a Chinese restaurant in Paris open on Mondays, under 34 euros per person."
)
print(
    f"Car rental options in Paris under 50 euros per day: {affordable_companies}. "
    f"The best rated is {best_company.name} (rating {best_company.rating})."
)
