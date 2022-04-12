import os

for i, file in os.listdir():
    if "image" not in file or ".jpg" not in file:
        continue
    cat = str(int(file[6:-4])//80+1)
    if i % 5 == 0:
        os.replace(file, os.path.join("dataset", "flowers", "val", cat, file))
    else:
        os.replace(file, os.path.join("dataset", "flowers", "train", cat, file))

