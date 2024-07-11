
import firebase_admin
import requests
from firebase_admin import credentials

cred = credentials.Certificate("sorour-d61ef-firebase-adminsdk-8dpf1-790769a138.json")
firebase_admin.initialize_app(cred)

response = requests.get("https://sorour-d61ef-default-rtdb.firebaseio.com/definitions")

print(response.content)