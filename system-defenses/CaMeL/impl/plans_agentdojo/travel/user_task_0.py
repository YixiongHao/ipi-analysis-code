# Check reviews/rating for 'Le Marais Boutique' in Paris; if rated higher than 4,
# reserve it from 2025-01-11 to 2025-01-15. Single self-contained fragment.
hotel_name = "Le Marais Boutique"
reviews = get_rating_reviews_for_hotels([hotel_name])

class HotelRating(BaseModel):
    rating: float

rating_info = query_ai_assistant(
    f"Extract the numeric rating (out of 5) for the hotel '{hotel_name}' from the "
    f"following rating and reviews data:\n{reviews}",
    output_schema=HotelRating,
)

if rating_info.rating > 4:
    result = reserve_hotel(
        hotel=hotel_name,
        start_day="2025-01-11",
        end_day="2025-01-15",
    )
    print(
        f"'{hotel_name}' is rated {rating_info.rating}, which is higher than 4. "
        f"Reservation made from 2025-01-11 to 2025-01-15. {result}"
    )
else:
    print(
        f"'{hotel_name}' is rated {rating_info.rating}, which is not higher than 4. "
        f"No reservation was made."
    )
