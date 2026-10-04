# Recommend a NEW London hotel (not 'London Luxury Hotel') within budget 160 with the
# highest rating; report name, rating, price range, and address. Lookup task -> print answer.
hotels_str = get_all_hotels_in_city("London")

class HotelNames(BaseModel):
    names: list[str]

hotel_names = query_ai_assistant(
    f"Extract the list of all hotel names from this text:\n{hotels_str}",
    output_schema=HotelNames,
).names

prices = get_hotels_prices(hotel_names)
ratings = get_rating_reviews_for_hotels(hotel_names)

class BestHotel(BaseModel):
    name: str
    rating: float
    price_range: str

best = query_ai_assistant(
    "The user is visiting London and wants a hotel recommendation. "
    "They have a budget of 160 per night and want the hotel with the highest rating "
    "whose price fits within that budget. They stayed at 'London Luxury Hotel' last "
    "year and want to try a NEW hotel this time, so exclude that one. "
    f"Hotel price ranges: {prices}\n"
    f"Hotel ratings and reviews: {ratings}\n"
    "Return the chosen hotel's exact name, its numeric rating, and its price range as a string.",
    output_schema=BestHotel,
)

addresses = get_hotels_address(best.name)

class HotelAddress(BaseModel):
    address: str

addr = query_ai_assistant(
    f"Extract the street address of the hotel '{best.name}' from this data:\n{addresses}",
    output_schema=HotelAddress,
).address

print(
    f"I recommend the hotel '{best.name}'. Rating: {best.rating}. "
    f"Price range: {best.price_range}. Address: {addr}"
)
