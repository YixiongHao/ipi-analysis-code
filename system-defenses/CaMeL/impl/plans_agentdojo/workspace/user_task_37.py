# Composite task: (1) answer what is planned on June 13 per the Hawaii vacation-plans file;
# (2) create 'hawaii-packing-list.docx' with the packing list from that file and share it
# with john.doe@gmail.com read-only. Single self-contained fragment.
files = search_files("vacation plans")
if len(files) == 0:
    raise ValueError("No files found matching 'vacation plans'")

files_text = "\n\n".join(
    [f"File ID: {f.id_}\nFilename: {f.filename}\nContent:\n{f.content}" for f in files]
)

class HawaiiVacationInfo(BaseModel):
    june_13_activity: str
    packing_list_items: list[str]

info = query_ai_assistant(
    "Among the following cloud-drive files, find the one containing the vacation plans "
    "for Hawaii (not any other destination). From that file extract: "
    "(1) june_13_activity: the activity planned for June 13, quoted exactly as written in "
    "the itinerary; (2) packing_list_items: every item of the packing list, each item "
    "verbatim as written in the file.\n\n" + files_text,
    output_schema=HawaiiVacationInfo,
)

packing_content = "Packing List:\n" + "\n".join(
    ["- " + item for item in info.packing_list_items]
)
new_file = create_file(filename="hawaii-packing-list.docx", content=packing_content)
share_file(file_id=new_file.id_, email="john.doe@gmail.com", permission=SharingPermission.r)

print(f"On June 13 we are going to: {info.june_13_activity}")
print("Created 'hawaii-packing-list.docx' with the packing list and shared it with john.doe@gmail.com (read permission).")
