import pymongo
import smtplib
from twilio.rest import Client
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import time
import base64
from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()
# Set up MongoDB connection
client = pymongo.MongoClient(os.getenv("MONGODB_URL"))
db = client["object_detection_db"]
collection = db["detections"]


# Twilio credentials
account_sid = os.getenv('TWILIO_ACCOUNT_SID')
auth_token = os.getenv('TWILIO_AUTH_TOKEN')
twilio_number = os.getenv('TWILIO_NUMBER')
target_number = os.getenv('TARGET_NUMBER')

#Send Email
def send_email(subject, body, image_base64):
    from_email = 'sdw4@gmail.com'  # Your email address
    to_email = 'wee70@gmail.com'   # Recipient's email address

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Attach the image directly as MIMEImage
    image_data = base64.b64decode(image_base64)
    image = MIMEImage(image_data)
    msg.attach(image)

    # Connect to the SMTP server and send the email
    smtp_server = os.getenv('SMTP_SERVER')
    smtp_port = int(os.getenv('SMTP_PORT'))
    smtp_username = os.getenv('SMTP_USERNAME')
    smtp_password = os.getenv('SMTP_PASSWORD')

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
        print("Email sent")




#message
def send_sms(message):
    client = Client(account_sid, auth_token)
    message = client.messages.create(
        body=message,
        from_=twilio_number,
        to=target_number
    )
    print("sms end")

last_id=None
while True:
    latest_document = collection.find({}).sort("_id", pymongo.DESCENDING).limit(1).next()
    latest_id = str(latest_document["_id"])


    if last_id != latest_id:
        last_id = latest_id
        latitude=latest_document["latitude"]
        longitude=latest_document["longitude"]
        image_data = latest_document["image_filename"]
        time_date=latest_document["timestamp"]
        address=latest_document["address"]
        # Retrieve and decode the image data
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        message = f"Weapon detected at Latitude: {latitude}, Longitude: {longitude}, Address: {address}, Time: {time_date}"

      # Send email
        send_email("Weapon Detected", message, image_base64)

      # Send SMS
        send_sms(message)
        print("Email and message sent")
    else:
        pass
    time.sleep(10)
