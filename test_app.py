from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import math
import os
import matplotlib.pyplot as plt


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def load_images(folder_path):
    return [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith((".png", ".jpg", ".jpeg"))]



def get_image_query_similarity(query, images_paths):
    images = [Image.open(img_path) for img_path in images_paths]
    inputs = processor(text=[query] * len(images), images=images, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    return logits_per_image.softmax(dim=0).tolist() # normalize




def get_similarity(query, image_path):
    image = Image.open(image_path)
    inputs = processor(text=query, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    # return normalize_score_with_sigmoid(logits_per_image)
    return logits_per_image.softmax(dim=1)


def normalize_score_with_sigmoid(logit_score):
    normalized_score = 1 / (1 + math.exp(-logit_score))
    return normalized_score


# query = "a mexican cat"
query = input("Enter a query: ")
folder_path = "./images"
image_paths = load_images(folder_path)

similarity_scores = get_image_query_similarity(query, image_paths)
best_match_index = similarity_scores.index(max(similarity_scores))
best_image_path = image_paths[best_match_index]

best_image = Image.open(best_image_path)
plt.imshow(best_image)
plt.axis('off')
# plt.title(f"Best Match: {os.path.basename(best_image_path)}\nScore: {similarity_scores[best_match_index]:.4f}")
plt.show()


# similarity = get_similarity(query, "cat_on_couch.jpg")
# print("Similarity Score: ", similarity)
