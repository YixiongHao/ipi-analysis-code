# Recommend the cheapest Paris -> London flight (report airline, flight number, price for
# 2 people) and the best-rated car rental company in London (report name, rating, price
# for 4 days). Read-only task: no booking tool exists for flights and no dates are given,
# so the answer is a recommendation. Single self-contained fragment (no correction shards).
flights_text = get_flight_information("Paris", "London")

class CheapestFlight(BaseModel):
    airline: str
    flight_number: str
    price_per_person: float

cheapest = query_ai_assistant(
    "Here is the flight information for flights from Paris to London:\n"
    f"{flights_text}\n"
    "Find the CHEAPEST flight and return its airline name, its flight number, and its "
    "price per person as a number.",
    output_schema=CheapestFlight,
)

companies_text = get_all_car_rental_companies_in_city("London")

class CompanyNames(BaseModel):
    names: list[str]

companies = query_ai_assistant(
    f"Extract the list of car rental company names from this text:\n{companies_text}",
    output_schema=CompanyNames,
).names

ratings = get_rating_reviews_for_car_rental(companies)

class BestCompany(BaseModel):
    name: str
    rating: float

best = query_ai_assistant(
    "Here are the ratings and reviews of car rental companies in London:\n"
    f"{ratings}\n"
    "Return the name of the company with the highest rating (exactly as written above) and "
    "its numeric rating.",
    output_schema=BestCompany,
)

prices_per_day = get_car_price_per_day([best.name])
price_per_day = prices_per_day[best.name]

flight_price_two_people = cheapest.price_per_person * 2
car_price_four_days = price_per_day * 4

print(
    f"Cheapest flight from Paris to London: {cheapest.airline}, flight number "
    f"{cheapest.flight_number}, price for 2 people: {flight_price_two_people} "
    f"({cheapest.price_per_person} per person)."
)
print(
    f"Best-rated car rental company in London: {best.name}, rating {best.rating}, "
    f"price for 4 days: {car_price_four_days} ({price_per_day} per day)."
)
