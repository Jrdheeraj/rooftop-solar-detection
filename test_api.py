import requests

url = "http://localhost:8000/predict"

image_path = "data/processed/images_all/906.0.jpg"

session = requests.Session()
session.trust_env = False  # ignore system/college proxy settings

with open(image_path, "rb") as f:
    files = {"file": (image_path, f, "image/jpeg")}
    data = {"confidence": "0.5"}
    response = session.post(url, files=files, data=data)

print("Status code:", response.status_code)
print("Response JSON:")
print(response.text)
