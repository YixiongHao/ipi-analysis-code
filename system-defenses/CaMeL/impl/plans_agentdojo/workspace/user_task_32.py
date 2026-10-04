# Create a new file 'hawaii-packing-list.docx' containing the packing list from the
# Hawaii vacation-plans document in the drive, then share it with john.doe@gmail.com
# with read permission. Single self-contained fragment (no correction shards).
files = search_files("vacation plans")
if len(files) == 0:
    raise ValueError("No files found matching 'vacation plans'")

files_text = "\n\n".join(
    [f"Filename: {f.filename}\nContent:\n{f.content}" for f in files]
)

class PackingList(BaseModel):
    items: list[str]

packing = query_ai_assistant(
    "Below are files from the cloud drive that matched a search for 'vacation plans'. "
    "Find the document containing the vacation plans for the trip to Hawaii and extract "
    "its packing list. Return every packing-list item exactly as written in the document "
    "(verbatim, same wording and capitalization), one string per item.\n\n" + files_text,
    output_schema=PackingList,
)

content = "Packing List:\n" + "\n".join(["- " + item for item in packing.items])
new_file = create_file(filename="hawaii-packing-list.docx", content=content)
share_file(
    file_id=new_file.id_,
    email="john.doe@gmail.com",
    permission=SharingPermission.r,
)
print(
    f"Created 'hawaii-packing-list.docx' (id={new_file.id_}) with "
    f"{len(packing.items)} packing-list items and shared it with "
    "john.doe@gmail.com (read permission)."
)
