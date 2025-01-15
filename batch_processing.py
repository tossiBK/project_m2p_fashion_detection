import os
import requests
import pandas as pd
import shutil
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# configurations variables. need to be adjusted depending on how the final folders will be setup
# local folders as it takes the images from there, but can be adjusted as needed and be from a cloud drive for example
BATCH_FILES_FOLDER = "batch_images/"
PENDING_FOLDER = os.path.join(BATCH_FILES_FOLDER, "pending/")
PROCEEDED_FOLDER = os.path.join(BATCH_FILES_FOLDER, "proceeded/")
OUTPUT_CSV = "output/predictions.csv"
API_URL = "http://127.0.0.1:5001/invocations"  # API endpoint locally. in a cloud / hoster environment, it need use a loadbalancer endpoint for example
CHUNK_SIZE = 50

# ensure output folders exist to avoid abortion if not setup correctly
os.makedirs(PENDING_FOLDER, exist_ok=True)
os.makedirs(PROCEEDED_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# define image preprocessing tranfsormation. This must match the one from the training and what we have in the API endpoint
# TODO maybe we need to bring it out in a shared file, so we can avoid issues here, but thats optimization
transform = Compose([
    Resize((128, 128)),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def get_images(folder):
    """ 
    get list of image paths in the specified folder. right ow we use only jpg by the dataset, but here we show already it can be easily extended in future
    """
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def move_images_to_pending():
    """
    move the first CHUNK_SIZE images from the root folder to the pending folder.
    """
    root_images = get_images(BATCH_FILES_FOLDER)
    chunk = root_images[:CHUNK_SIZE]
    for img in chunk:
        shutil.move(img, os.path.join(PENDING_FOLDER, os.path.basename(img)))
    return len(chunk)

def log_remaining_images():
    """
    log the number of images left in the root folder and calculate remaining chunks. This is only for debug and comfort to see the progression in a possible long running task
    can also be used for monitoring.
    """
    root_images = get_images(BATCH_FILES_FOLDER)
    remaining_images = len(root_images)
    remaining_chunks = (remaining_images + CHUNK_SIZE - 1) // CHUNK_SIZE  # Round up
    print(f"{remaining_images} images left in the root folder.")
    print(f"Approximately {remaining_chunks} chunks remaining.")

def preprocess_images(image_paths):
    """
    preprocess images for API consumption.
    """
    instances = []
    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        tensor = transform(image).tolist() 
        instances.append(tensor)
    return instances

def send_to_api(instances):
    """
    send preprocessed images to the API and return predictions.
    """
    response = requests.post(API_URL, json={"instances": instances})
    if response.status_code == 200:
        return response.json()  # expected to return predictions list. it may need som e more error handling later
    else:
        response.raise_for_status()

def process_images():
    """
    process images in chunks. would suggest 50 as setup. So if something is, we have not to fix all items and only at max 50 wrong elements.add()
    too less items may end in overhead in API calls, too many may also lead to much data send at once
    """
    processed_count = 0

    # initialize the output CSV
    if not os.path.exists(OUTPUT_CSV):
        pd.DataFrame(columns=["filename", "prediction"]).to_csv(OUTPUT_CSV, index=False)

    while True:
        # log remaining images and chunks
        log_remaining_images()

        # move images from root to pending, but only if empty, otherwise use those first
        # TODO: we need to make all if this later Thread save (multiple instances running on it)
        if not get_images(PENDING_FOLDER):
            moved_count = move_images_to_pending()
            if moved_count == 0:
                print("All images have been processed.")
                break

        # get images in the pending folder
        images = get_images(PENDING_FOLDER)

        # process images in chunks
        chunk = images[:CHUNK_SIZE]
        print(f"Processing {len(chunk)} images...")

        try:
            # preprocess images
            instances = preprocess_images(chunk)

            # send images to API and get predictions
            predictions = send_to_api(instances)

            # log predictions
            results = [{"filename": os.path.basename(img), "prediction": pred} for img, pred in zip(chunk, predictions)]
            results_df = pd.DataFrame(results)
            results_df.to_csv(OUTPUT_CSV, mode='a', header=False, index=False)

            # move images to proceeded folder, so we won´t touch them again
            # possible also to delete, but for security reasons better to have them a certain time and clean up later
            for img in chunk:
                shutil.move(img, os.path.join(PROCEEDED_FOLDER, os.path.basename(img)))

            processed_count += len(chunk)
            print(f"Chunk processed. Total images processed: {processed_count}")

        except Exception as e:
            print(f"Error processing chunk: {e}")
            # TODO aborting for now, but may better only to log issues and warn
            # so process can run to the end if a minor issue only happen
            break

if __name__ == "__main__":
    process_images()
