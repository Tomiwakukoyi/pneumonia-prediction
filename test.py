import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


# Load the trained model
model = load_model("pneumonia_cnn_model.h5")  
print("✅ Model loaded successfully.")

# Folder containing test images
val_dir = "val"

# Scan both subdirectories (NORMAL & PNEUMONIA)
for category in ["NORMAL", "PNEUMONIA"]:
    category_path = os.path.join(val_dir, category)

    # Ensure the category folder exists
    if not os.path.exists(category_path):
        print(f"⚠️ Skipping: {category_path} (Folder not found)")
        continue

    print(f"\n📂 Scanning folder: {category}")

    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)

        # Ensure it's an image file
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"⚠️ Skipping non-image file: {img_name}")
            continue

        print(f"🖼️ Processing image: {img_name}")

        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize

        # Predict
        prediction = model.predict(img_array)[0][0]  

        # Print the result
        if prediction > 0.5:
            print(f"🔴 {img_name} → PNEUMONIA")
        else:
            print(f"🟢 {img_name} → NORMAL")

input("\n✅ Done! Press Enter to exit...")
