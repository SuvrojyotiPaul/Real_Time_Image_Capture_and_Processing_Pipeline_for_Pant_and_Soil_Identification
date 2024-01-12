
import sys
print(sys.executable)
print(sys.version)

import argparse
import os
import time
from typing import Dict, List

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from google.cloud import storage

HEIGHT, WIDTH = 1024, 1024

import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'plantandsoil-10958c09a47d.json'


# this function label_map is used to read a file that contains mappings of numerical IDs (labels) to their corresponding names.

def label_map(label_path: str) -> Dict[str, str]: #It takes a string as an argument which is the path to the label map file.
    """Process the label map file."""
    return_label_map = {} #It creates an empty dictionary called return_label_map. This will be used to store the mappings of labels to names.
    with open(label_path, "r") as label_file: #It opens the file specified by the label_path in read mode.
        for line in label_file: #It goes through each line in the file. For each line, it checks if the string "id" is in the line.
            if "id" in line: #If "id" is in the line, it means this line contains a label.
                label_index = int(line.split(":")[-1]) #then splits the line on the ":" character and takes the last element, which should be the label index. It converts this to an integer.
                label_name = next(label_file).split(":")[-1].strip().strip(
                    "'\"")  #It then reads the next line in the file which should contain the corresponding name for the label. It splits this line on the ":" character, strips any leading or trailing whitespace, and removes any single or double quotes.
                return_label_map[int(label_index)] = label_name #It then adds this label and name to the return_label_map dictionary.
    return_label_map[0] = "Background" #Once all lines have been processed, it sets the label for index 0 to be "Background". This is a common convention in object detection tasks where the background class is represented by 0.
    return return_label_map #returns the return_label_map dictionary which now contains mappings of labels to names.

#The function get_instance_mask is converting what is a segmentation mask from the model's output into an instance mask for a specific class.
def get_instance_mask(msk: np.ndarray, lab: int, thresh: int) -> np.ndarray: #
    """Convert class mask to instaance mask"""
    instance_mask = np.zeros_like(msk, np.uint8) #This line creates a new array of the same shape as msk (which is model's segmentation mask), but filled with zeros. This is done to initialize the instance mask. The np.uint8 specifies that the new array should have the data type of 8-bit unsigned integer
    instance_mask[np.where(msk > thresh)] = lab + 1 #Here, the function is finding where in msk the values are greater than thresh. This is basically selecting the pixels in the mask that have a high enough confidence score. For these pixels, it is setting their corresponding value in instance_mask to be lab + 1
    return instance_mask


def download_blob(bucket_name, source_blob_name, destination_file_name): #download_blob function is used to download a file (also known as a blob) from a Google Cloud Storage bucket
    """Downloads a blob from the bucket."""
    storage_client = storage.Client() #creates a storage_client which is an object that allows you to interact with Google Cloud Storage.
    bucket = storage_client.bucket(bucket_name) #uses the storage_client to get a reference to the bucket specified by bucket_name.
    blob = bucket.blob(source_blob_name) #gets a reference to the blob in the bucket specified by source_blob_name.

    blob.download_to_filename(destination_file_name) #downloads the blob to a file with the name destination_file_name on your local machine.
    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name  #downloads the blob to a file with the name destination_file_name on your local machine.
        )
    )


class Predictor:  #The Predictor class defined here is used for making predictions with a machine learning mode

    def __init__(self, model, height, width, out_name, thresh):  #is the class constructor. It sets up the Predictor object with the model, dimensions of the input image (height and width), output name and threshold value.
        self.model = model 
        self.model_height = height
        self.model_width = width
        self.output_name = out_name
        self.threshold = thresh

    def feed_forward(self, instance: List[np.ndarray]) -> Dict:  #method is part of the Predictor class, and it's responsible for making predictions with the model
        """Single feed-forward pipeline""" #a list of numpy arrays representing an image (or multiple images).
        result = self.model(inputs=instance) # It calls the model (which was stored in self.model in the __init__ method) with the input instance. This is where the model actually makes a prediction based on the input. The result is stored in the result variable.
        return result # It then returns the prediction result. Dictionary with keys: "output_1", "output_2", "output_3", and "output_4"

    def preprocess(self, input_image: np.ndarray) -> np.ndarray: #  function prepares the input image before it's fed to the model for prediction
        """Preprocess the image array before it is fed to the model"""
        model_input = Image.fromarray(input_image).convert("RGB") # This line first converts the input numpy array to a PIL Image object, then it converts the image to RGB format if it's not already.
        model_input = model_input.resize((self.model_width, self.model_height)) # resizes it to the dimensions the model expects
        model_input = np.array(model_input).astype(np.float32) # This line first converts the PIL Image object back into a numpy array, and then it converts the datatype of that array to float32. This is done because the model expects the input data to be in this format.
        model_input = np.expand_dims(model_input, 0) # and then adds an extra dimension at the start (which is a common requirement for models that expect a batch of images). This is done because the model expects a batch of images as input, even if we're only processing one image at a time. The extra dimension represents the batch size.
        return model_input

    def postprocess(self, model_output: List[np.ndarray]) -> List: #method in the Predictor class takes the output from the machine learning model and processes it to a more usable form.
        """Postprocess the model output""" # The model's output is expected to be a dictionary with keys: "output_1", "output_2", "output_3", and "output_4". The values associated with these keys are extracted and converted to numpy arrays, which are then assigned to sco, classes, boxes, and masks respectively.
        sco = np.array(model_output["output_1"][0]) #represents scores associated with each prediction made by the model.
        classes = np.array(model_output["output_2"][0]).astype(np.int16) # represents the class labels that the model predicted.
        boxes = np.array(model_output["output_3"][0]) #represent the bounding box coordinates for detected objects in the image.
        masks = np.array(model_output["output_4"][0]) #represent the segmentation masks for detected objects in the image.
        _filter = np.where(sco > self.threshold) # This line creates a filter, an array of indices where the predicted scores (sco) are greater than a set threshold (self.threshold)
        sco = sco[_filter] 
        classes = classes[_filter] #These lines apply the filter to the scores, classes, boxes, and masks. After these lines, sco, classes, boxes, and masks only contain elements where the score was above the threshold.
        boxes = boxes[_filter]
        masks = masks[_filter]

        masks_output = []  
        for cls, each_mask in zip(classes, masks): #The next part of the code processes each mask in the masks array
            output_mask = get_instance_mask(each_mask, cls, self.threshold) #for each mask, it calls the get_instance_mask function, which converts the class mask to an instance mask. 
            masks_output.append(output_mask) #These masks are stored in the masks_output list.

        if masks_output: # If the masks_output list is not empty, it is converted to a numpy array. If it is empty, it is set to None
            masks_output = np.stack(masks_output) 
        else:
            masks_output = None

        return [boxes, masks_output, sco, classes] #a list containing the boxes, masks_output, scores, and classes is returned.

    def predict(self, image: np.ndarray) -> Dict: #is the main function that brings together all the steps for making a prediction using the provided model.
        """Send an image for prediction, then process it to be returned
           
            afterwards.
        """
        preprocessed = self.preprocess(image)
        predicted = self.feed_forward(preprocessed)
        postprocessed = self.postprocess(predicted)
        return postprocessed


parser = argparse.ArgumentParser(  #This code is defining command-line arguments for a Python program using the argparse library. 
    prog="Datature Instance Segmentation Tensorflow Predictor",
    description="Predictor to Predict Instance Segmentation Tensorflow Model.")
parser.add_argument("-i", "--input_folder_path")
parser.add_argument("-o", "--output_folder_path")
parser.add_argument("-m", "--model_path")
parser.add_argument("-l", "--label_map_path")
parser.add_argument("-t", "--threshold")
parser.add_argument("-b", "--bucket_name")

if __name__ == "__main__": #The code inside this if-statement only runs if the script is being run directly.
    args = parser.parse_args()  #this line parses command-line arguments. 
    input_path = args.input_folder_path 
    output_path = args.output_folder_path   #The next few lines extract the values of the command-line arguments into variables
    model_path = args.model_path
    label_map_path = args.label_map_path
    threshold = float(args.threshold)
    bucket_name = args.bucket_name

    loaded = tf.saved_model.load(model_path) #This line loads a saved TensorFlow model from the path specified in the model_path argument.
    loaded_model = loaded.signatures["serving_default"] #In TensorFlow, a model can have multiple 'signatures', which are different ways the model can be used. By default, when you save a TensorFlow model, it will have a 'serving' signature that is used to make predictions. In TensorFlow, a model can have multiple 'signatures', which are different ways the model can be used.
    output_name = list(loaded_model.structured_outputs.keys())[0] #The output layer's name can be particularly important, especially when the model has multiple outputs. it extracts the name of the first output layer from the model. structured_outputs is a dictionary where each key-value pair is the name of an output layer and the corresponding TensorFlow Tensor for that output.
    predictor = Predictor(loaded_model, HEIGHT, WIDTH, output_name, threshold) #This line creates an instance of the Predictor class, using the loaded model, some predefined height and width, the output name, and the threshold.

    color_map = {
        1: [0, 255, 0],  # Green for plant  #This line creates a dictionary that maps class numbers (1 and 2) to colors (green and yellow). This map is used for visualization of the results.
        2: [255, 255, 0]  # Yellow for soil
    }

    category_map = label_map(label_map_path) #This line is calling the function label_map with an argument label_map_path

    storage_client = storage.Client() #This line creates a new client to interact with Google Cloud Storage. This client allows the program to perform operations like uploading and downloading files from Google Cloud Storage.
    bucket = storage_client.get_bucket(bucket_name) #This line gets a reference to a specific storage bucket within Google Cloud Storage. The name of this bucket is specified by the variable bucket_name

    while True: #This is an infinite loop. It will keep running until it's manually stopped or some condition within the loop triggers a break or return.
        blobs = bucket.list_blobs() #This line retrieves a list of all the blobs (files) in the specified bucket.

        for blob in blobs:#This loop goes through each file in the bucket one by one
            if ".jpg" in blob.name or ".png" in blob.name: # If the file is an image file (either JPEG or PNG), it proceeds with the following actions.
                download_blob(bucket_name, blob.name, os.path.join(input_path, blob.name)) #This function call downloads the image file from the bucket and saves it to the local file system in a directory specified by input_path
                print("\nPredicting for", blob.name) 

                img = Image.open(os.path.join(input_path, blob.name)).convert("RGB") #it opens the downloaded image and converts it into an RGB image.
                img_array = np.array(img) #This line converts the image into a NumPy array, which is a format that can be processed by the model.
                bboxes, output_masks, scores, labels = predictor.predict(img_array) #this line feeds the image to the model for prediction. The model returns bounding boxes, masks (binary images that cover detected objects), scores, and labels .

                if len(bboxes) != 0 and output_masks is not None: #This condition checks whether the model has detected any objects in the image.
                    for each_bbox, label, score in zip(bboxes, labels, scores): #This condition checks whether the model has detected any objects in the image.
                        color = color_map.get(label) #It retrieves the color that corresponds to the class of the detected object.

                        # Draw bounding box
                        cv2.rectangle( #It uses the OpenCV library to draw a rectangle (bounding box) around the detected object on the original image. The color of the rectangle is determined by the class of the object. The dimensions and position of the rectangle are specified by each_bbox.
                            img_array,
                            (
                                int(each_bbox[1] * img_array.shape[0]),  #, each_bbox[1] and each_bbox[0] are the relative coordinates (between 0 and 1), Multiplying each of these by the height (img_array.shape[0]) and width (img_array.shape[1]) of the image, respectively, gives the actual location in pixels
                                int(each_bbox[0] * img_array.shape[1]),
                            ),
                            (
                                int(each_bbox[3] * img_array.shape[0]),
                                int(each_bbox[2] * img_array.shape[1]),
                            ),
                            color,
                            2, #thickness of the rectangle
                        )

                        # # Draw label background
                        # cv2.rectangle(
                        #     img_array,
                        #     (
                        #         int(each_bbox[1] * img_array.shape[0]),
                        #         int(each_bbox[2] * img_array.shape[1]),
                        #     ),
                        #     (
                        #         int(each_bbox[3] * img_array.shape[0]),
                        #         int(each_bbox[2] * img_array.shape[1] + 15),
                        #     ),
                        #     color,
                        #     -1,
                        # )

                        # Insert label class & score
                        mid_x = int((each_bbox[3] + each_bbox[1]) * 0.5 * img_array.shape[0])
                        mid_y = int((each_bbox[2] + each_bbox[0]) * 0.5 * img_array.shape[1])  #This part of the code is calculating the middle point (x, y coordinates) of the bounding box

                        cv2.putText( #This is a function call to place some text on the image. The image is stored in the variable img_array
                            img_array,
                            "Class: {}, Score: {}".format( #This is the text that will be placed on the image. It's saying something like "Class: [class name], Score: [score]"
                                str(category_map[label]),
                                str(round(score, 2)),      #category_map[label] gets the class name associated with the current label, and round(score, 2) rounds the score to 2 decimal places.
                            ),
                            (mid_x, mid_y), #These are the coordinates where the text will be placed on the image.
                            cv2.FONT_HERSHEY_SIMPLEX, # This is the font style that the text will be in.
                            0.6, #This is the size of the text.
                            (255, 255, 255), #This is the color of the text, in RGB format
                            2, #This is the thickness of the text.
                            cv2.LINE_AA, #this stands for Anti Aliased line type which makes the text look smoother.
                        )

                    img_mask = Image.fromarray(img_array.astype(np.uint8))  #This is converting the img_array (which is an array of pixel values) back into an image that can be saved to a file. The array is converted to unsigned 8-bit integers because that's a common format for pixel values (each pixel value is between 0 and 255).
                    specific_output_path = os.path.join(output_path, blob.name) #This is constructing a file path where the image will be saved. The file path is made by joining the output_path (a directory) and blob.name (the name of the file).
                    img_mask.save(specific_output_path) #This is saving the image to a file at the location specified by specific_output_path.
                    print("Prediction saved to", specific_output_path) #This is printing a message to let you know where the image was saved.

                else:
                    print("No detections for", blob.name) #If there were no detections in the image (i.e., if len(bboxes) == 0 or output_masks is None), this will print a message saying that there were no detections for this image.

                # delete the blob after processing
                # blob.delete()

        # Wait for 3 minutes
        time.sleep(180)
