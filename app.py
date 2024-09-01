import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils
import time
from twilio.rest import Client
import PIL.Image
import PIL.ExifTags
import os
import threading
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import pymongo
import subprocess
import geocoder
import reverse_geocoder
from geopy.geocoders import Nominatim
import requests
import pywhatkit

#Date Time
import datetime 
datetime= datetime.datetime.now()
detected_time=0
time_date=f"{datetime.day}:{datetime.month}:{datetime.year},{datetime.hour}:{datetime.minute}:{datetime.second}"



# Set up MongoDB connection
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["object_detection_db"]
collection = db["detections"]


#file location
CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'


#File Path
paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
    'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
    'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
    'PROTOC_PATH':os.path.join('Tensorflow','protoc')
 }

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}




#Send Email
def send_email(subject, body, image_filename):
    from_email ='@gmail.com'  # Your email address
    to_email = '@gmail.com'   # Recipient's email address

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Attach the captured image
    with open(image_filename, 'rb') as img_file:
        image = MIMEImage(img_file.read())
        msg.attach(image)

    # Connect to the SMTP server and send the email
    smtp_server = 'smtp.com'  # Gmail's SMTP server address
    smtp_port = 587
    smtp_username = '@gmail.com'  # Your email address
    smtp_password = ''   

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.sendmail(from_email, to_email, msg.as_string())
        print("Email sent successfully!")
    except Exception as e:
        print("Error sending email:", e)
    finally:
        server.quit()
        print("email send")

#send whatapp
def whatapp(image_filename):
    img=image_filename
    time_hour=datetime.hour
    time_min=datetime.minute
    ph_no=""

    pywhatkit.sendwhatmsg(ph_no,img, time_hour, time_min)
    

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-22')).expect_partial()

#Detection
@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections
category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])


def get_ip_location():
    try:
        ip_add = requests.get("https://get.geojs.io/v1/ip.json")
        ip_address = ip_add.json()['ip']
        
        url = 'https://get.geojs.io/v1/ip/geo/' + ip_address + '.json'
        geo_request = requests.get(url)
        geo_data = geo_request.json()
        
        latitude ='28.706930' # geo_data.get('latitude', '')
        longitude = '77.135000' #geo_data.get('longitude', '')
        city = geo_data.get('city', '')
        region = geo_data.get('region', '')
        country = geo_data.get('country', '')
        postal_code = geo_data.get('area_code', '')  # Use 'area_code' as postal code
        
        #address_parts = [city, region, country, postal_code]
        #address = ', '.join(part for part in address_parts if part)
        address="Pitampura, New Delhi, India"
        return latitude, longitude, ip_address, address
    except Exception as e:
        print(f"Error: {e}")
        return None




#camera
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
capture_interval = 1 * 60  # 4 minutes in seconds
capture_last_time = time.time()  # Initialize the last capture time

while cap.isOpened():
    ret, frame = cap.read()
    #ime.sleep(2)
    image_np = np.array(frame)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'] + label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=1,
        min_score_thresh=0.65,
        agnostic_mode=False)

    cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

    current_time = time.time()
    
    if np.any(detections['detection_scores'] > 0.95) and current_time>detected_time:
        detected_time=current_time+(1*60)
        image_filename = f"images/detected_image_{current_time}.jpg"
        cv2.imwrite(image_filename, image_np)
        with open(image_filename, "rb") as f:
            image_data = f.read()
        
   
        # Get GPS location from the image
        try:
            latitude,longitude,ip_address,address = get_ip_location()
    
            if latitude is not None and longitude is not None:
                location_message = f" Latitude:{latitude}, Decimal Longitude:{longitude} wapon detected"
                
                email_subject = "Object Detected"
                email_body = location_message
                current_time=time.time()
                #exit file in of the 
                #half of the data
                 
                # Store data in MongoDB
                detection_data = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "image_filename": image_data,
                    "timestamp": time_date,
                    "ip_address":ip_address,
                    "address":address
                }
                collection.insert_one(detection_data)
                
            else:
                print("GPS information not available or error occurred.")
        except (AttributeError, KeyError, ValueError) as e:
            print("Error extracting GPS information:", e)

        
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
    
    
    
    
