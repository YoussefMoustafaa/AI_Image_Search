from pinecone import Pinecone
from keys import PINECONE_API_KEY
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import streamlit as st


pinecone = Pinecone(api_key=PINECONE_API_KEY)

index_name = "fashion"
index = pinecone.Index(index_name)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

folder_path = "./images"

def generate_embedding(image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    embedding = model.get_image_features(**inputs).detach().numpy().flatten()
    return embedding

image_data = []
for file_name in os.listdir(folder_path):
    if file_name.endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(folder_path, file_name)
        embedding = generate_embedding(image_path)
        vector = {
            "id": file_name,
            "values": embedding.tolist(),
            "metadata": {"file_name": file_name}
        }
        image_data.append(vector)

index.upsert(vectors=image_data, namespace= "ns1")

def generate_query_embedding(query):
    inputs = processor(text=[query], return_tensors="pt", padding=True)
    return model.get_text_features(**inputs).detach().numpy().flatten()


st.title("AI Image Search for Retail")

query = st.text_input("What are you looking for? ")

if query:
    with st.spinner("Searching for images..."):
        query_embedding = generate_query_embedding(query)

        results = index.query(
            namespace="ns1",
            vector=query_embedding.tolist(),
            top_k=5,
            include_values=True,
            include_metadata=True
        )

        st.subheader("Results:")
        for result in results['matches']:
            file_name = result['metadata']['file_name']
            score = result['score']

            image_path = os.path.join(folder_path, file_name)
            image = Image.open(image_path)
            st.image(image, caption=f"Score: {score:.4f}", use_column_width=True)
        
