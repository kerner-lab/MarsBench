import os
import zipfile

import wget

DATASET = "DoMars16K"
# Create DoMars16k Dataset
URL = "https://zenodo.org/records/1048301/files/hirise-map-proj.zip?download=1"

zip_file_path = f"/home/kkasodek/MarsBench/datasets/DeepMars"

if not (os.path.isdir(zip_file_path)):
    os.mkdir(zip_file_path)

# Downloading Dataset
wget.download(URL, zip_file_path)

# Unzip if required
for zipped_folder in os.listdir(zip_file_path):
    if zipped_folder.endswith((".zip")):
        zipped_folder_path = os.path.join(zip_file_path, zipped_folder)
with zipfile.ZipFile(zipped_folder_path, "r") as zip:
    zip.extractall(path=zip_file_path)
    zip.close()
