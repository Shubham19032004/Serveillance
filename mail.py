import pymongo
import smtplib
from twilio.rest import Client
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import time
import base64
# Set up MongoDB connection
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["object_detection_db"]
collection = db["detections"]


# Twilio credentials
account_sid = 'ACae74c03577f0e0c567d2d7f868e99683'
auth_token = '84cd4df53b27363e7990028c3ef2e5ca'
twilio_number = '+12622879836'
target_number = '+918368227176'

#Send Email
def send_email(subject, body, image_base64):
    from_email = 'shubhamnainwal4@gmail.com'  # Your email address
    to_email = 'shubhamnainwal70@gmail.com'   # Recipient's email address

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
    smtp_server = 'smtp.gmail.com'  # Gmail's SMTP server address
    smtp_port = 587
    smtp_username = 'shubhamnainwal4@gmail.com'  # Your email address
    smtp_password = 'bzcvvwibvpaxpjop'   

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
