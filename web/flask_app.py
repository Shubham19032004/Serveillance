from flask import Flask, render_template
from pymongo import MongoClient
from datetime import datetime
from dateutil import tz
import base64

app = Flask(__name__)

# Configure MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client.object_detection_db  
detections = db.detections

@app.route('/')
def index():
    # Fetch data from the database
    data = list(detections.find())

    # Convert binary image data to base64-encoded strings
    for item in data:
        image_binary = item['image_filename']
        image_base64 = base64.b64encode(image_binary).decode('utf-8')
        item['image_filename'] = image_base64

    return render_template('index.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)
