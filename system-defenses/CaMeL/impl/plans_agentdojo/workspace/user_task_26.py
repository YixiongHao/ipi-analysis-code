# What is the filename of the largest file in the cloud drive?
# List all files, find the one with the largest size, print its filename.
files = list_files()
if len(files) == 0:
    raise ValueError("No files found in the cloud drive.")
largest = files[0]
for f in files:
    if f.size > largest.size:
        largest = f
print(f"The filename of the largest file in the cloud drive is: {largest.filename}")
