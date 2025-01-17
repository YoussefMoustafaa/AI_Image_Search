import requests
import streamlit as st
from PIL import Image
import torch
from transformers import CLIPModel, CLIPTokenizer, CLIPImageProcessor
from io import BytesIO
from keys import API_KEY, CSE_ID, URL



model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")


def google_image_search(query):
    params = {
        "q": query,
        "cx": CSE_ID,
        "key": API_KEY,
        "searchType": "image",
        "num": 10
    }
    response = requests.get(URL, params)
    response.raise_for_status()
    results = response.json()
    return [(item["link"], item["title"]) for item in results.get("items", [])]


st.title("AI Image Search")

query = st.text_input("What's on your mind? ")
if query:
    try:
        images = google_image_search(query)
        for image_url, title in images:
            st.image(image_url, caption=title)
    except Exception as e:
        st.error(f"Error fetching images: {e}")



def download_and_compare_images(query, image_urls):
    images = [Image.open(BytesIO(requests.get(url).content)) for url in image_urls]
    inputs = {
        "input_ids": tokenizer([query] * len(images), return_tensors="pt", padding=True)["input_ids"],
        "pixel_values": image_processor(images, return_tensors="pt")["pixel_values"]
    }    
    outputs = model(**inputs)
    scores = outputs.logits_per_image.softmax(dim=0).tolist()
    return sorted(zip(image_urls, scores), key=lambda x: x[1], reverse=True)


if query:
    try:
        ranked_images = download_and_compare_images(query, images)
        for url, score in ranked_images[:5]:
            st.image(url, caption=score)
    except Exception as e:
        st.error(f"Error: {e}")

    

