import os
import shutil
import kagglehub
import gdown
import zipfile

# 1. Download dataset with kagglehub
print("Descargando dataset Kaggle...")
dataset_path = kagglehub.dataset_download("gpiosenka/70-dog-breedsimage-data-set")

# 2. Move train folders to data/images
current_dir = os.getcwd()


train_dir = os.path.join(dataset_path, "train")
target_dir = os.path.join(current_dir, "data", "images")
os.makedirs(target_dir, exist_ok=True)

print("Moviendo carpetas...")
for breed_folder in os.listdir(train_dir):
    full_path = os.path.join(train_dir, breed_folder)
    if os.path.isdir(full_path):
        shutil.move(full_path, os.path.join(target_dir, breed_folder))

# 3. Remove everything except data/images
print("Limpiando directorio de trabajo...")
shutil.rmtree(dataset_path)

# 4. Download and extract chroma DB from Google Drive
print("Descargando Chroma DB desde Google Drive...")
gdown.download(
    "https://drive.google.com/file/d/1oKGQLc4MZd8mDt8iheLk80kmeeaS3t6B/view?usp=sharing",
    "chroma.zip",
    quiet=False,
    fuzzy=True
)

print("Extrayendo Chroma DB...")
with zipfile.ZipFile("chroma.zip", "r") as zip_ref:
    zip_ref.extractall("data/chroma")

os.remove("chroma.zip")


print("✅ Proceda.")