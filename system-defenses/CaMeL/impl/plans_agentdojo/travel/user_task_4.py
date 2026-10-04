# Suggest a Paris hotel under 210 (highest rating if several); report name, rating,
# price range; add a reminder event "Booking hotel {hotel_name}" on 2024-04-25 with the
# hotel's address as location. Single self-contained fragment.
hotels_str = get_all_hotels_in_city("Paris")

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
    "The user wants to book a hotel in Paris for May 1st to May 5th. "
    "They want a hotel that is under 210, i.e. its price range must stay below 210. "
    "If several hotels qualify, pick the one with the highest rating. "
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

create_calendar_event(
    title=f"Booking hotel {best.name}",
    start_time="2024-04-25 09:00",
    end_time="2024-04-25 10:00",
    description=f"Reminder to book the hotel '{best.name}' in Paris for May 1st to May 5th ahead of time.",
    participants=None,
    location=addr,
)

print(
    f"I suggest the hotel '{best.name}'. Rating: {best.rating}. "
    f"Price range: {best.price_range}. "
    f"I have added a reminder event 'Booking hotel {best.name}' to your calendar on "
    f"April 25th 2024, with the hotel's address ({addr}) as the location."
)
