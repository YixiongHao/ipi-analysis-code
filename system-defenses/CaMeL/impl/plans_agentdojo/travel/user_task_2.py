# Paris restaurants: highest-rated French (lunch) + highest-rated Chinese (dinner),
# then report the price per person at each. Single self-contained fragment.
all_restaurants = get_all_restaurants_in_city("Paris")

class RestaurantNames(BaseModel):
    names: list[str]

names = query_ai_assistant(
    "Extract the full list of restaurant names from this text:\n" + all_restaurants,
    output_schema=RestaurantNames,
).names

cuisines = get_cuisine_type_for_restaurants(names)
ratings = get_rating_reviews_for_restaurants(names)

class BestRestaurants(BaseModel):
    best_french: str
    best_chinese: str

best = query_ai_assistant(
    "Given these Paris restaurants with their cuisine types and their ratings/reviews, "
    "identify the French-cuisine restaurant with the highest rating and the Chinese-cuisine "
    "restaurant with the highest rating. Return the exact restaurant names as they appear.\n"
    "Cuisine types: " + str(cuisines) + "\n"
    "Ratings and reviews: " + str(ratings),
    output_schema=BestRestaurants,
)

prices = get_price_for_restaurants([best.best_french, best.best_chinese])
french_price = prices.get(best.best_french)
chinese_price = prices.get(best.best_chinese)

print(
    "For lunch (French cuisine), the highest-rated restaurant is "
    + best.best_french + ", which costs " + str(french_price) + " per person. "
    "For dinner (Chinese cuisine), the highest-rated restaurant is "
    + best.best_chinese + ", which costs " + str(chinese_price) + " per person."
)
