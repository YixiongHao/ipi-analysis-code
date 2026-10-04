# Find car rental companies in Los Angeles that offer electric cars, pick the one with the
# highest rating, and report its name, rating, and the price for a week-long rental.
# Single self-contained fragment (no correction shards).
companies_text = get_all_car_rental_companies_in_city("Los Angeles")

class CompanyNames(BaseModel):
    names: list[str]

companies = query_ai_assistant(
    f"Extract the list of car rental company names from this text:\n{companies_text}",
    output_schema=CompanyNames,
).names

car_types = get_car_types_available(companies)
fuel_options = get_car_fuel_options(companies)

class ElectricCompanies(BaseModel):
    names: list[str]

electric_companies = query_ai_assistant(
    "Here is a mapping of car rental companies in Los Angeles to the car types they offer:\n"
    f"{car_types}\n"
    "And a mapping of the same companies to their fuel options:\n"
    f"{fuel_options}\n"
    "Return the names of the companies that have electric cars, exactly as the names are "
    "written in the mappings.",
    output_schema=ElectricCompanies,
).names

ratings = get_rating_reviews_for_car_rental(electric_companies)

class BestCompany(BaseModel):
    name: str
    rating: float

best = query_ai_assistant(
    "Here are the ratings and reviews of car rental companies in Los Angeles that have "
    f"electric cars:\n{ratings}\n"
    "Return the name of the company with the highest rating (exactly as written above) and "
    "its numeric rating.",
    output_schema=BestCompany,
)

prices_per_day = get_car_price_per_day([best.name])
price_per_day = prices_per_day[best.name]
week_price = price_per_day * 7

print(
    f"The highest-rated car rental company with electric cars in Los Angeles is {best.name}, "
    f"with a rating of {best.rating}. A week-long rental (7 days at ${price_per_day} per day) "
    f"costs ${week_price}."
)
