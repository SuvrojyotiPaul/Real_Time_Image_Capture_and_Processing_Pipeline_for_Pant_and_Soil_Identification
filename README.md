# Real_Time_Image_Capture_and_Processing_Pipeline_for_Pant_and_Soil_Identification
## Development of a Real-Time Image Capture and Processing Pipeline for Plant  and Soil Identification in Agricultural Fields

<p style="font-size:14px;">
This project aimed at creating a real-time image capture and processing pipeline for field 
environments using Raspberry Pi cameras and a Mask R-CNN model. The successful 
execution of the project demonstrated the potential of integrating Internet of Things (IoT) 
devices with machine learning for real-time field environment monitoring. The pipeline 
provided updated snapshots of the field, which could be viewed and evaluated through a 
web interface.
</p>

<p style="font-size:14px;">
 This project's primary motivation was to address the prevailing challenges in real-time 
agricultural monitoring, which often involves laborious and time-consuming manual 
surveillance. Developing an automated system can greatly enhance the monitoring 
efficiency, enabling faster detection of any anomalies and thereby aiding in timely 
intervention. In this regard, the implementation of machine learning models for image 
analysis is instrumental, as they can accurately identify and classify various elements in the 
field, such as differentiating between plant and soil regions
</p>

## Overall, this project aspired to integrate hardware and software components into a functioning system capable of delivering real-time agricultural field monitoring.

# For a deeper understanding of the project and the whole process, please refer to the report pdf already available in the repository, or use this link:
- [Image Capture and Processing Pipeline Development for Field Environment Analysis](https://github.com/SuvrojyotiPaul/Real_Time_Image_Capture_and_Processing_Pipeline_for_Pant_and_Soil_Identification/blob/main/Image%20Capture%20and%20Processing%20Pipeline%20Development%20for%20Field%20Environment%20Analysis.pdf)

 <br><br>

## This is the the folder structure of the local computer
![Screenshot 2023-05-25 015803](https://github.com/SuvrojyotiPaul/Real_Time_Image_Capture_and_Processing_Pipeline_for_Pant_and_Soil_Identification/assets/122437351/c9347563-0cb8-4743-af70-a4d877c7ae57)

<br><br>

## This is the the folder structure of each raspberry pis
![Screenshot 2023-05-23 114822](https://github.com/SuvrojyotiPaul/Real_Time_Image_Capture_and_Processing_Pipeline_for_Pant_and_Soil_Identification/assets/122437351/e5b3f761-644e-4974-89de-54d565ef98a6)

<br><br>

## THis is how the web interface looks like with the mask rcnn output
![Screenshot 2023-05-24 103824](https://github.com/SuvrojyotiPaul/Real_Time_Image_Capture_and_Processing_Pipeline_for_Pant_and_Soil_Identification/assets/122437351/00db63b2-8186-4208-99b2-a2a57c134bc6)

<br><br>

## This how the model metrics look in the web interface, for more detail about the metrics and evaluation please check the report pdf mentioned above
![Screenshot 2024-01-13 130203](https://github.com/SuvrojyotiPaul/Real_Time_Image_Capture_and_Processing_Pipeline_for_Pant_and_Soil_Identification/assets/122437351/2c196ddf-5f43-4248-b892-19c0f5ee985d)

<br></br>

## To reproduce the Results:-
- 1) Start the myscriptmodified.py file located in the local (/home/pi/Desktop/Test) of each R-pi via terminal.
- 2) Go to the google cloud storage account to monitor the uploaded file (optional)
- 3) Go to the local computer and navigate to the project folder
- 4) Now you have to run the predictmodified.py to do that open cmd/terminal while you are in that folder and put the following :
  
python3 predictmodified.py -i "D:\Datature\509980f8-9b43-4e7c-9aac-9e8260b0f9b7-ckpt-21-tensorflow (1)\input" -o 
"D:\Datature\509980f8-9b43-4e7c-9aac-9e8260b0f9b7-ckpt-21-tensorflow (1)\output" -m "D:\Datature\509980f8-9b43-4e7c9aac-9e8260b0f9b7-ckpt-21-tensorflow (1)\saved_model" -l "D:\Datature\509980f8-9b43-4e7c-9aac-9e8260b0f9b7-ckpt-21-
tensorflow (1)\label_map.pbtxt" -t 0.5 -b "plantrasp

Replace the paths accordingly

- 5) After that navigate to the input folder to see the downloaded images uploaded to the google cloud from r-pis and also go to the output folder (optional)
- 6) Then open the app.py and run it , it will then generate the local web server which will show the output
