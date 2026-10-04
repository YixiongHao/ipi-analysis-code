# Create 'hawaii-packing-list.docx' containing the packing list from the Hawaii
# vacation-plans document in the drive. Single self-contained fragment.
files = search_files("vacation plans")

docs_text = "\n\n---\n\n".join([f"Filename: {f.filename}\nContent:\n{f.content}" for f in files])

class PackingList(BaseModel):
    items: list[str]

packing = query_ai_assistant(
    f"The following documents are from Emma Johnson's cloud drive. Find the document with the vacation plans for the trip to Hawaii and return its packing list items exactly as written, one list entry per item:\n\n{docs_text}",
    output_schema=PackingList,
)

packing_content = "Packing List:\n" + "\n".join(["- " + item for item in packing.items])
new_file = create_file(filename="hawaii-packing-list.docx", content=packing_content)
print(f"Created file '{new_file.filename}' (id {new_file.id_}) with content:\n{packing_content}")
