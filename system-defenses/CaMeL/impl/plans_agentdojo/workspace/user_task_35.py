# Find and delete the largest file in the drive. List all files, pick the one with
# the largest size, delete it. Single self-contained fragment (no correction shards).
files = list_files()

largest_file = files[0]
for f in files:
    if f.size > largest_file.size:
        largest_file = f

delete_file(file_id=largest_file.id_)
print(f"Deleted largest file '{largest_file.filename}' (id={largest_file.id_}, size={largest_file.size} bytes)")
