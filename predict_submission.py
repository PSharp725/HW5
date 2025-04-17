import os
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import preprocess_input
from tqdm import tqdm

IMG_SIZE = (96, 96)  # or whatever you're using
BATCH_SIZE = 64
TEST_IMG_DIR = 'histopathologic-cancer-detection/test/'
MODEL_PATH = "final_model.keras"
OUTPUT_PATH = "submission.csv"

def load_and_preprocess(img_path, img_size):
    try:
        img = Image.open(img_path).convert("RGB").resize(img_size)
        img_array = np.array(img).astype("float32")
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        print(f"[ERROR] {img_path}: {e}")
        return None

def main():
    print("Loading model...")
    model = keras.models.load_model(MODEL_PATH)

    test_images_list = sorted(os.listdir(TEST_IMG_DIR))
    predictions = []

    print("Starting inference...")
    for i in tqdm(range(0, len(test_images_list), BATCH_SIZE)):
        batch_filenames = test_images_list[i:i+BATCH_SIZE]
        batch_imgs = []
        batch_ids = []

        for img_name in batch_filenames:
            img_path = os.path.join(TEST_IMG_DIR, img_name)
            img_array = load_and_preprocess(img_path, IMG_SIZE)
            if img_array is not None:
                batch_imgs.append(img_array)
                batch_ids.append(img_name.replace(".tif", ""))

        if batch_imgs:
            batch_imgs = np.stack(batch_imgs)
            preds = model.predict(batch_imgs).flatten()
            print("Batch min:", preds.min(), "max:", preds.max(), "mean:", preds.mean())
            predictions.extend(zip(batch_ids, preds))

    print(model.summary())

    submission_df = pd.DataFrame(predictions, columns=["id", "label"])
    submission_df["label"] = submission_df["label"].astype("float32")
    submission_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved predictions to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()