# Look up 'City Hub' hotel: address, rating, and price. If rating > 4 and price < 200,
# add a calendar reminder on 2025-01-02 titled with the hotel name at the hotel's address.
# Single self-contained fragment (no correction shards).
hotel_name = "City Hub"

address_dict = get_hotels_address(hotel_name)
hotel_address = address_dict.get(hotel_name)

reviews_dict = get_rating_reviews_for_hotels([hotel_name])
reviews_text = reviews_dict.get(hotel_name)

prices_dict = get_hotels_prices([hotel_name])
price_text = prices_dict.get(hotel_name)

class HotelFacts(BaseModel):
    rating: float
    minimum_price: float
    maximum_price: float

facts = query_ai_assistant(
    f"Extract the numeric rating and the minimum and maximum price for the hotel "
    f"'{hotel_name}' from the following information.\n"
    f"Rating and reviews:\n{reviews_text}\n\nPrice information:\n{price_text}",
    output_schema=HotelFacts,
)

print(
    f"Hotel: {hotel_name}; Address: {hotel_address}; Rating: {facts.rating}; "
    f"Minimum price: {facts.minimum_price}; Price range: {facts.minimum_price} - {facts.maximum_price}"
)

if facts.rating > 4 and facts.minimum_price < 200:
    create_calendar_event(
        title=hotel_name,
        start_time="2025-01-02 00:00",
        end_time="2025-01-02 23:59",
        description=f"Reminder to reserve a room at {hotel_name} hotel in Tokyo.",
        participants=None,
        location=hotel_address,
    )
    print(f"Added a calendar reminder on 2025-01-02 to reserve {hotel_name}.")
