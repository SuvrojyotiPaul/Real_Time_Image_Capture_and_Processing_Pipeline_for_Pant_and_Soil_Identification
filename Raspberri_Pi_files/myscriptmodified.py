import os
import time
import cv2
from datetime import datetime
from google.cloud import storage

# Set Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/pi/Desktop/Test/google_cloud_key.json'

# Google Cloud Storage parameters
bucket_name = 'plantrasp'
storage_client = storage.Client()

# Create a folder to store the images
output_folder = "webcam_images"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to capture and save image
def capture_image(output_folder, image_name):
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera
    if not cap.isOpened():
        print("Error: Camera not found.")
        return

    ret, frame = cap.read()
    if ret:
        save_path = os.path.join(output_folder, f"{image_name}.jpg")
        if os.path.exists(save_path):
            os.remove(save_path)
        cv2.imwrite(save_path, frame)
    else:
        print("Error: Cannot capture image.")

    cap.release()
    cv2.destroyAllWindows()

# Function to upload a file to Google Cloud Storage
def upload_to_gcs(source_file_path, destination_blob_name):
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_path)

# Function to delete a file from Google Cloud Storage
def delete_from_gcs(blob_name):
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.delete()

# Main loop to capture 5 images every 12 seconds
previous_images = []
while True:
    current_images = []
    for i in range(5):
        image_name = f"rasp1_{i}"
        capture_image(output_folder, image_name)
        file_path = os.path.join(output_folder, f"{image_name}.jpg")
        blob_name = f"{image_name}.jpg"
        upload_to_gcs(file_path, blob_name)
        current_images.append(blob_name)
        time.sleep(12)  # Wait 12 seconds between each image

    time.sleep(120)  # Wait 2 minutes before capturing the next set of images

    # Delete previous images from Google Cloud Storage
    for blob_name in previous_images:
        delete_from_gcs(blob_name)

    previous_images = current_images

