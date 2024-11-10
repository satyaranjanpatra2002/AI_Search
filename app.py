#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load the data
courses_df = pd.read_csv('Data_free_course.csv')

# Initialize the language model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-mpnet-base-v2')

model = load_model()

# Function to retrieve the top recommendations
def get_recommendations(query, top_k=5):
    # Encode the query and course titles
    query_embedding = model.encode(query, convert_to_tensor=True)
    course_embeddings = model.encode(courses_df['Title'].tolist(), convert_to_tensor=True)
    
    # Calculate cosine similarity
    similarities = util.pytorch_cos_sim(query_embedding, course_embeddings)
    
    # Get top k results with scores
    top_results = similarities[0].topk(k=top_k)
    indices = top_results.indices.cpu().numpy()  # Convert to a numpy array
    scores = top_results.values.cpu().numpy()    # Convert to a numpy array
    
    return indices, scores

# Streamlit UI
st.title("🔍 Smart Course Search Tool")
st.write("Discover the best free courses tailored to your interests!")

# User input
user_query = st.text_input("🔎 What course are you looking for? (e.g., Python, Machine Learning)")

# Styling improvements
st.markdown("""
    <style>
    .course-container {
        border: 1px solid #ddd;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 8px;
        background-color: #f9f9f9;
    }
    .course-title {
        color: #1f77b4;
        font-size: 20px;
        font-weight: bold;
    }
    .course-detail {
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

if user_query:
    st.write("### Top Course Recommendations:")
    
    # Get top recommendations based on the user query
    indices, scores = get_recommendations(user_query)
    
    for idx, score in zip(indices, scores):
        course = courses_df.iloc[idx]  # Use integer indexing for iloc
        with st.container():
            st.markdown(f"""
                <div class="course-container">
                    <div class="course-title"><a href="{course.get('Link', '#')}" target="_blank">{course['Title']}</a></div>
                    <div class="course-detail"><strong>Duration:</strong> {course.get('Duration', 'N/A')}</div>
                    <div class="course-detail"><strong>Lessons:</strong> {course.get('Lessons', 'N/A')}</div>
                    <div class="course-detail"><strong>Description:</strong> {course.get('Description', 'No description available')}</div>
                    <div class="course-detail"><strong>Relevance Score:</strong> {score:.2f}</div>
                </div>
            """, unsafe_allow_html=True)


# In[ ]:




