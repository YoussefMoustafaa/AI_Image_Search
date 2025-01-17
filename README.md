# AI-Powered Image Search App
## Overview

This application uses AI to create an intelligent image search system. Users can input a query (text description), and the app retrieves and displays the most relevant images from a pre-stored database. Powered by Pinecone for vector search and OpenAI's CLIP model for generating image and text embeddings, this app demonstrates the potential of AI in retail, e-commerce, and media management.

## Features

* AI-Driven Image Search: Finds images based on natural language descriptions.
* Pinecone Integration: Stores image embeddings for fast and accurate vector-based searches.
* Streamlit Interface: Provides a simple and interactive web-based UI for query inputs and result visualization.

## Use Cases

* Retail and E-commerce: Allow users to search for products by describing them in natural language.
* Creative Inspiration: Find relevant images for projects or marketing materials based on descriptive prompts.

## Setup Instructions
1. Prerequisites

* Python 3.8 or higher
* Pinecone API Key
* Install required Python packages:

```
pip install pinecone torch transformers pillow streamlit
```

2. Setting Up Pinecone

    Sign up for a Pinecone account and get your API key.
    Create a Pinecone index in the website.
    Initialize the index:

    pc = Pinecone(api_key="YOUR_API_KEY")
    index = pc.Index(index_name)

3. Running the App

    Clone or download the repository.
    Place your images in the images/ folder.
    Generate embeddings for your images and upload them to Pinecone:

```
python embeddings.py
```

Run the Streamlit app:

```
    streamlit run app.py
```

    Access the app in your browser at http://localhost:8501.

## How It Works

1. Embedding Generation:
    * The app uses OpenAI's CLIP model to generate 512-dimensional embeddings for images and text.
    * These embeddings represent the semantic meaning of the content.

2. Vector Storage:
    * Pinecone stores image embeddings as vectors.
    * Each vector is indexed and associated with metadata, like file names or categories.

3. Search and Retrieval:
    * When a user inputs a query, its embedding is generated using CLIP.
    * Pinecone performs a similarity search to find the closest matching image embeddings.

4. Result Display:
    * The app retrieves the metadata and displays the corresponding images.
