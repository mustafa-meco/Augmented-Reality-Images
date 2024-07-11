import requests
import numpy as np
import cv2

def send_image(image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    response = requests.post("http://localhost:5000/augment", data=image_data, headers={'Content-Type': 'application/octet-stream'})
    return response.content

def send_image(image, url):
    # Encode the image to binary format
    _, image_binary = cv2.imencode(".jpg", image)
    image_binary = image_binary.tobytes()

    # Send the image to the API endpoint
    response = requests.post(url, data=image_binary)

    # Check the response status code to see if the request was successful
    if response.status_code != 200:
        print("Failed to send image:", response.text)
        return None
    
    return response.content

def decode_response(response):
    nparr = np.frombuffer(response, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

def get_image(url):
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to get image:", response.text)
        return None
    return response.content

imgOut = decode_response(get_image("http://localhost:5000/get-image"))
cv2.imshow("img", imgOut)
cv2.waitKey(0)



# cap = cv2.VideoCapture(2)
# while True:
#     success, img = cap.read()
#     imgOut = decode_response(send_image(img, "http://localhost:5000/augment"))


#     cv2.imshow("IMage", imgOut)
#     cv2.waitKey(1)