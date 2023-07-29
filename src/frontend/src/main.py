
"""
You can find a list of emojis at: https://raw.githubusercontent.com/omnidan/node-emoji/master/lib/emoji.json
"""
import streamlit as st
from utils import *
import json 

st.set_page_config(
    page_title=" HealthyGamer Transcripts",
    # page_icon=" :blue_book",  # EP: how did they find a symbol?
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Healthy Gamer GG Transcripts!")
st.header("Welcome to the Healthy Gamer GG Demo")
st.write(
    "Enter a query about HealthyGamerGG Videos. You can check out the original channel [here](https://www.youtube.com/@HealthyGamerGG). Your query will be answered using his video transcriptions as context, using embeddings from sentence-transformers/multi-qa-MiniLM-L6-cos-v1."
)
text = st.text_input("Query text:", value="example: What are common signs of depression?", label_visibility='hidden')
if text != "example: What are common signs of depression?":
    resp = get_results(text)
    if resp.status_code == 200:
        data = json.loads(resp.text)
        texts, titles, webps, urls = parse_results(data=data)
        thumbnails = [parse_webp(i) for i in webps]
        
        #--- Main Loop --- 
        num_images = len(thumbnails)
        num_columns = 3
        num_rows = (num_images - 1) // num_columns + 1

        count = 0
        for row in range(num_rows):
            col1, col2, col3 = st.columns(num_columns)
            for col, image_url in zip([col1, col2, col3], thumbnails[row * num_columns: (row + 1) * num_columns]):
                col.image(image_url, use_column_width=True, caption=f'{titles[row]}')
                # Add a text area input for each image
                markdown = f"""[{texts[count]}]({urls[count]})"""
                user_text = col.markdown(markdown)
                count += 1
    else: 
        st.write("Sorry Query Couldn't be completed")

#TODO: Add in UI Header
#TODO: Check Data to make sure this makes sense
#TODO: Format Data so that the same videos do not show up


