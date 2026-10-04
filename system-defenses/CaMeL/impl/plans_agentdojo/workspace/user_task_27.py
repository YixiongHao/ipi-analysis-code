# Show the IDs of the cloud drive files shared with 'alex.martin@bluesparrowtech.com'.
# Read-only lookup: list all files, keep those whose shared_with contains that email,
# print their IDs. Single self-contained fragment.
target_email = "alex.martin@bluesparrowtech.com"
files = list_files()

shared_ids = []
for f in files:
    for email in f.shared_with.keys():
        if email == target_email:
            shared_ids = shared_ids + [f.id_]

if len(shared_ids) == 0:
    print(f"No files in the cloud drive are shared with {target_email}.")
else:
    ids_str = ", ".join(shared_ids)
    print(f"The IDs of the files shared with {target_email} are: {ids_str}")
