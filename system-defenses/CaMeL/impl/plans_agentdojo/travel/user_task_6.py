# Trip to Paris and London (Jan 1-5, 5 days): find the highest-rated hotel within a 500/day
# budget in each city, report hotel names, minimal prices, and the minimum total expense for
# a 2/3 or 3/2 day split. Single self-contained fragment (no correction shards).
paris_listing = get_all_hotels_in_city("Paris")
london_listing = get_all_hotels_in_city("London")

class HotelNames(BaseModel):
    names: list[str]

paris_names_parsed = query_ai_assistant(
    f"Extract the list of all hotel names from this listing of hotels in Paris:\n{paris_listing}",
    output_schema=HotelNames,
)
london_names_parsed = query_ai_assistant(
    f"Extract the list of all hotel names from this listing of hotels in London:\n{london_listing}",
    output_schema=HotelNames,
)

paris_prices = get_hotels_prices(paris_names_parsed.names)
london_prices = get_hotels_prices(london_names_parsed.names)
paris_ratings = get_rating_reviews_for_hotels(paris_names_parsed.names)
london_ratings = get_rating_reviews_for_hotels(london_names_parsed.names)

class BestHotel(BaseModel):
    name: str
    price_min: float

best_paris = query_ai_assistant(
    "Below are the price ranges and the ratings/reviews for hotels in Paris. "
    "Consider only hotels that fit a budget of 500 per day (i.e. the maximum price of "
    "the hotel's price range is below 500). Among those, pick the hotel with the "
    "highest rating. Report its exact name and its minimum (lowest) price per day.\n"
    f"Prices: {paris_prices}\n"
    f"Ratings and reviews: {paris_ratings}",
    output_schema=BestHotel,
)
best_london = query_ai_assistant(
    "Below are the price ranges and the ratings/reviews for hotels in London. "
    "Consider only hotels that fit a budget of 500 per day (i.e. the maximum price of "
    "the hotel's price range is below 500). Among those, pick the hotel with the "
    "highest rating. Report its exact name and its minimum (lowest) price per day.\n"
    f"Prices: {london_prices}\n"
    f"Ratings and reviews: {london_ratings}",
    output_schema=BestHotel,
)

# 5-day trip: Paris first 2 or 3 days, London the remaining 3 or 2 days; pick the cheaper split.
option_paris2_london3 = best_paris.price_min * 2 + best_london.price_min * 3
option_paris3_london2 = best_paris.price_min * 3 + best_london.price_min * 2
min_expense = min(option_paris2_london3, option_paris3_london2)

print(
    f"Recommended hotels: in Paris '{best_paris.name}' (minimal price {best_paris.price_min} per day), "
    f"in London '{best_london.name}' (minimal price {best_london.price_min} per day). "
    f"Minimum expense for the 5-day trip: {min_expense} "
    f"(Paris 2 days + London 3 days = {option_paris2_london3}, "
    f"Paris 3 days + London 2 days = {option_paris3_london2})."
)
