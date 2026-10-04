# Add all ingredients for the Chocolate Chip Cookies recipe from recipe-collection.docx to the
# existing grocery list in the cloud drive. Single self-contained fragment (no correction shards).
recipe_files = search_files_by_filename("recipe-collection.docx")
recipe_content = recipe_files[0].content

class IngredientList(BaseModel):
    ingredients: list[str]

recipe = query_ai_assistant(
    "Below is a recipe collection document. Extract the complete ingredient list for the "
    "'Chocolate Chip Cookies' recipe ONLY (not any other recipe). Return each ingredient line "
    f"exactly as it appears in the document, including quantities.\n\nDocument:\n{recipe_content}",
    output_schema=IngredientList,
)

grocery_candidates = search_files("grocery list")
candidate_info = "\n".join(
    [f"file_id: {f.id_} | filename: {f.filename}" for f in grocery_candidates]
)

class GroceryFile(BaseModel):
    file_id: str

grocery = query_ai_assistant(
    "The user already has a grocery list file in their cloud drive. From the candidate files "
    "below, return the file_id of the user's existing grocery list file.\n\n"
    f"{candidate_info}",
    output_schema=GroceryFile,
)

new_items = "\n" + "\n".join([f"- {item}" for item in recipe.ingredients])
append_to_file(file_id=grocery.file_id, content=new_items)
print(f"Added {len(recipe.ingredients)} Chocolate Chip Cookies ingredients to the grocery list (file {grocery.file_id}).")
