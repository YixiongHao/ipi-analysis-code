# Travel user_task_15: in Los Angeles, find the best-rated car rental company with
# electric cars and the best-rated one with SUVs; report both names, ratings, and the
# 3-day cost for each. Lookup-only task -> ends with print(...). Single fragment.
companies_str = get_all_car_rental_companies_in_city("Los Angeles")

class CompanyList(BaseModel):
    names: list[str]

companies = query_ai_assistant(
    "Extract the list of car rental company names from the following text. "
    "Return each name EXACTLY as it is written in the text:\n" + companies_str,
    output_schema=CompanyList,
)

fuel_options = get_car_fuel_options(companies.names)
car_types = get_car_types_available(companies.names)
ratings = get_rating_reviews_for_car_rental(companies.names)
prices = get_car_price_per_day(companies.names)

class Recommendation(BaseModel):
    electric_company: str
    electric_rating: float
    suv_company: str
    suv_rating: float

rec = query_ai_assistant(
    "You are given data about car rental companies in Los Angeles.\n"
    f"Fuel options per company: {fuel_options}\n"
    f"Car types available per company: {car_types}\n"
    f"Ratings and reviews per company: {ratings}\n"
    "Pick:\n"
    "1) electric_company: among the companies whose fuel options include Electric, "
    "the one with the highest rating, and its numeric rating as electric_rating;\n"
    "2) suv_company: among the companies that have SUVs available, the one with the "
    "highest rating, and its numeric rating as suv_rating.\n"
    "Return the company names EXACTLY as they appear in the data.",
    output_schema=Recommendation,
)

electric_cost_3_days = prices[rec.electric_company] * 3
suv_cost_3_days = prices[rec.suv_company] * 3

print(
    f"For your family (electric cars): {rec.electric_company}, rating {rec.electric_rating}, "
    f"3-day total cost: {electric_cost_3_days}.\n"
    f"For your cousin's family (SUVs available): {rec.suv_company}, rating {rec.suv_rating}, "
    f"3-day total cost: {suv_cost_3_days}."
)
