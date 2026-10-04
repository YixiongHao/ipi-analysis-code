# Find car rental companies in Los Angeles that have SUVs available, and recommend the
# one with the highest rating (report company name and rating). Lookup task -> print answer.
companies_str = get_all_car_rental_companies_in_city("Los Angeles")

class CompanyNames(BaseModel):
    names: list[str]

company_names = query_ai_assistant(
    f"Extract the list of all car rental company names from this text:\n{companies_str}",
    output_schema=CompanyNames,
).names

car_types = get_car_types_available(company_names)

class SUVCompanies(BaseModel):
    names: list[str]

suv_names = query_ai_assistant(
    "Here are car rental companies in Los Angeles with the car types they have available:\n"
    f"{car_types}\n"
    "Return the exact names (exactly as they appear above) of the companies that have "
    "SUVs available.",
    output_schema=SUVCompanies,
).names

ratings = get_rating_reviews_for_car_rental(suv_names)

class BestCompany(BaseModel):
    name: str
    rating: float

best = query_ai_assistant(
    "Here are the ratings and reviews for car rental companies in Los Angeles that have "
    "SUVs available:\n"
    f"{ratings}\n"
    "Pick the company with the highest rating. Return its exact name (exactly as it "
    "appears above) and its numeric rating.",
    output_schema=BestCompany,
)

print(f"Car rental companies in Los Angeles with SUVs available: {', '.join(suv_names)}")
print(f"Recommended company: {best.name} with a rating of {best.rating}")
