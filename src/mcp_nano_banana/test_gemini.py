import json
import os
import base64
import google.generativeai as genai
from dotenv import load_dotenv
import requests

load_dotenv()

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-2.5-flash-image')

prompt = "Create nano-sized banana in a lab setting."
response = model.generate_content([prompt])
response = response.to_dict()

bytes_data = response["candidates"][0]["content"]["parts"][-1]["inline_data"]["data"]

generated_img = base64.b64decode(bytes_data)
with open('edited_nano_banana.png', 'wb') as out:
    out.write(generated_img)


# Upload image to Imgbb.host
# --- STEP 1: Ensure 'edited_nano_banana.png' exists and is a valid, non-empty image ---
try:
    with open('edited_nano_banana.png', 'rb') as image_file:
        # Read the binary data and encode it to a Base64 string
        generated_img_b64 = base64.b64encode(image_file.read()).decode('utf-8')
    
    if not generated_img_b64:
        raise ValueError("The image file 'edited_nano_banana.png' is empty.")

except FileNotFoundError:
    raise FileNotFoundError("The image file 'edited_nano_banana.png' was not found. Please ensure it exists.")


# --- STEP 2: Build and send the correct POST request to ImgBB ---
IMGBB_API_KEY = os.getenv("IMGBB_API_KEY")
if not IMGBB_API_KEY:
    raise ValueError("IMGBB_API_KEY environment variable not set or .env file is missing.")

upload_url = "https://api.imgbb.com/1/upload"

# All parameters go into the 'data' payload for the POST request
payload = {
    "key": IMGBB_API_KEY,
    "image": generated_img_b64,  # The Base64 string is the 'image' field
    "name": "nano_banana.png"   # Optional: specify a name for the file
}

try:
    print("Uploading image to ImgBB...")
    # Use the data= parameter, NOT files=
    resp = requests.post(upload_url, data=payload, timeout=60) # Increased timeout for larger files
    
    resp.raise_for_status() # Raise an error for bad status codes (4xx or 5xx)

    resp_json = resp.json()

    # ImgBB's success indicator is the 'data' key in the response
    if "data" not in resp_json:
        raise Exception(f"Imgbb upload failed. Response: {resp_json}")

    uploaded_url = resp_json["data"]["url"]
    print(f"Success! Image uploaded to {uploaded_url}")

except requests.exceptions.HTTPError as err:
    print(f"HTTP error occurred: {err}")
    print(f"Response body: {err.response.text}")
    raise
