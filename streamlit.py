# Imports
import streamlit as st
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
import requests
import os
#Update Comment for git
# Page styling
st.set_page_config(page_title="NLP", page_icon="📜", layout="wide")



# URL of the .bin file you want to download
url = 'https://ingridhansen.anthra.be/assets/files/pytorch_model.bin'

# Path where you want to save the downloaded file
local_filename = 'model/pytorch_model.bin'

def download_file(url, local_filename):
    # Send a HTTP request to the URL
    if os.path.exists(local_filename):
        print(f"File already exists: {local_filename}")
        return
    with requests.get(url, stream=True) as r:
        # Check if the request was successful
        if r.status_code == 200:
            # Open the local file in write-binary mode
            with open(local_filename, 'wb') as f:
                # Iterate over the response in chunks
                for chunk in r.iter_content(chunk_size=8192):
                    # Write each chunk to the local file
                    f.write(chunk)
            print(f"File downloaded successfully: {local_filename}")
        else:
            print(f"Failed to download file. HTTP Status Code: {r.status_code}")

# Call the function to download the file
download_file(url, local_filename)

# Containers for styling
header = st.container()
EDA = st.container()
Model = st.container()
with header:
    st.title("NLP: Toxic Comment Classification")
    st.write("We made a model that can classify toxic comments.")
    st.write("We will start by showing the EDA that we did.")

with EDA:
    st.header("Performing exploratory data analysis")
    col2, col3, col4 = st.columns(3)

    with st.expander("Cleaning the dataset", expanded=False):
        st.write('These are the first 7 lines in the original dataset:')
        data = pd.read_csv("./train.csv")
        st.write(data.head(7))
        st.write("As you can see there are multiple columns that specify what type of toxic comment it is. We decided that we will make one column which will specify if a comment is toxic or not, and no longer want to have multiple labels for that. We also dropped the 'id' column since we don't really need that.")
        df_toxic = data.copy()
        df_toxic['is_toxic'] = df_toxic[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].apply(lambda row: any(row), axis=1).astype(int)
        df_toxic = df_toxic.drop(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], axis=1)
        df_toxic = df_toxic.drop('id', axis=1)
        st.write(df_toxic.head(7))

    with st.expander("Distribution of toxic and not toxic comments", expanded=False):
        st.write(df_toxic['is_toxic'].value_counts())
        st.write("As we can see the dataset is heavily imbalenced. Let's make sure that this is fixed by reducing the number to 2000. We are undersampling.")
        df_toxic_balanced = pd.concat([
            df_toxic[df_toxic['is_toxic'] == 0].sample(2000, random_state=42),
            df_toxic[df_toxic['is_toxic'] == 1].sample(2000, random_state=42)
        ])

        df_toxic_balanced = df_toxic_balanced.sample(frac=1, random_state=42)

        st.write(df_toxic_balanced['is_toxic'].value_counts())   
    
    with st.expander("Renaming the label", expanded=False):
        st.write("We need to rename 'is_toxic' to 'label' for it to work with the model we want to use.")
        df_toxic_balanced = df_toxic_balanced.rename(columns={'is_toxic': 'label'})
        df_toxic_balanced['label'] = df_toxic_balanced['label'].astype(float)   
        st.write(df_toxic_balanced['label'].value_counts())  
        df_toxic_balanced['label'] = df_toxic_balanced['label'].map({0: 'Not Toxic', 1: 'Toxic'})


with Model:
        st.header("Classify your comment")
        # Load the model
        model_directory = 'NLPStreamlit/model'
        model = AutoModelForSequenceClassification.from_pretrained(model_directory, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model_directory, local_files_only=True)
        pipeline =  TextClassificationPipeline(model=model, tokenizer=tokenizer)
        col5, col6 = st.columns(2)
        with col5:
            # User enters a comment
            comment = st.text_area('Enter your comment here:')
            classify_button = st.button("Classify")
        with col6:
            if classify_button:
                st.write("")
                st.write("")
                st.write("")
                # Predict the comment and show what it predicted
                output = pipeline(comment)
                label = output[0]['label']
                score = output[0]['score'] * 100
                st.write('**Prediction**:', label)
                st.write('**Confidence**:', f"{score:.2f}%")

# Footer
footer="""<style>

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}


</style>
<div class="footer">
<p>Ingrid Hansen, Dina Boshnaq and Iris Loret - Big Data Group 6</p>
</div>


"""
st.markdown(footer,unsafe_allow_html=True)
    
