import streamlit as st
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline


# Function to load datasets
@st.cache_data
def load_data():
    customer_profiles = pd.read_csv("./customer_profiles.csv")
    demographic_details = pd.read_csv("./demographic_details.csv")
    transaction_history = pd.read_csv("./transaction_history.csv")
    social_media_activity = pd.read_csv("./social_media_activity.csv")

    # Merge all datasets on Customer_Id
    customers = customer_profiles.merge(demographic_details, on="Customer_Id", how="left") \
        .merge(transaction_history, on="Customer_Id", how="left") \
        .merge(social_media_activity, on="Customer_Id", how="left")

    return customers.fillna("")


# Function to load AI models
@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    return embedding_model, sentiment_analyzer


# Function to build FAISS customer index
@st.cache_resource
def build_faiss_index(_embedding_model, customers):
    customer_embeddings = []
    customer_ids = []

    for _, row in customers.iterrows():
        profile_text = f"{row['Age']} {row['gender']} {row['Interests']} {row['Preferences']} {row['Income per year']} " \
                       f"{row['Education']} {row['Occupation']} {row['Marital Status']} {row['Dependents']} " \
                       f"{row['Home Ownership']} {row['Nationality']} {row['Transaction Type']} {row['Category']}  " \
                       f"{row['Purchase_Date']} {row['Payment Mode']} {row['Merchant Name']} {row['Transaction Status']} {row['Tax Amount']} {row['Discount Applied']} " \
                       f"{row['Transaction Approval Status']} {row['Customer Segment']} {row['Platform']} {row['Content']} {row['Sentiment_Score']} {row['Intent']}"

        emb = _embedding_model.encode([profile_text])[0]
        customer_embeddings.append(emb)
        customer_ids.append(str(row["Customer_Id"]))

    customer_embeddings = np.array(customer_embeddings, dtype=np.float32)
    customer_index = faiss.IndexFlatL2(customer_embeddings.shape[1])
    customer_index.add(customer_embeddings)

    return customer_index, customer_ids


# Load data & models
customers = load_data()
embedding_model, sentiment_analyzer = load_models()
customer_index, customer_ids = build_faiss_index(embedding_model, customers)

customers["Customer_Id"] = customers["Customer_Id"].astype(str)

# Streamlit UI
st.title("AI-Powered Personalized Recommendation System")
st.write("Enter a valid Customer ID to receive personalized recommendations.")


customer_id = st.text_input("Enter your Customer ID").strip()

def getCustomerData(customer_id):
    if customer_id in customer_ids:
        #st.write("## Customer Details")
        user_profile = customers[customers["Customer_Id"] == customer_id].iloc[0]

        customer_info = {
            "Customer ID": user_profile["Customer_Id"],
            "Age": user_profile["Age"],
            "Gender": user_profile["gender"],
            "Interests": user_profile["Interests"],
            "Preferences": user_profile["Preferences"],
            "Income per year": user_profile["Income per year"],
            "Education": user_profile["Education"],
            "Occupation": user_profile["Occupation"],
            "Marital Status": user_profile["Marital Status"],
            "Dependents": user_profile["Dependents"],
            "Home Ownership": user_profile["Home Ownership"],
            "Nationality": user_profile["Nationality"],
            "Transaction Type": user_profile["Transaction Type"],
            "Category": user_profile["Category"],
            "Purchase Date": user_profile["Purchase_Date"],
            "Payment Mode": user_profile["Payment Mode"],
            "Merchant Name": user_profile["Merchant Name"],
            "Transaction Status": user_profile["Transaction Status"],
            "Tax Amount": user_profile["Tax Amount"],
            "Discount Applied": user_profile["Discount Applied"],
            "Transaction Approval Status": user_profile["Transaction Approval Status"],
            "Customer Segment": user_profile["Customer Segment"],
            "Platform": user_profile["Platform"],
            "Sentiment Score": user_profile["Sentiment_Score"],
            "Intent": user_profile["Intent"]
        }
        #st.table(pd.DataFrame(customer_info.items(), columns=["Attribute", "Value"]))

        # Create user profile embedding
        user_text = f"{user_profile['Age']} {user_profile['gender']} {user_profile['Interests']} {user_profile['Preferences']} {user_profile['Income per year']} " \
                    f"{user_profile['Education']} {user_profile['Occupation']} {user_profile['Marital Status']} {user_profile['Dependents']} " \
                    f"{user_profile['Home Ownership']} {user_profile['Nationality']} {user_profile['Transaction Type']} {user_profile['Category']}  " \
                    f"{user_profile['Purchase_Date']} {user_profile['Payment Mode']} {user_profile['Merchant Name']} {user_profile['Transaction Status']} {user_profile['Tax Amount']} {user_profile['Discount Applied']} " \
                    f"{user_profile['Transaction Approval Status']} {user_profile['Customer Segment']} {user_profile['Platform']} {user_profile['Content']} {user_profile['Sentiment_Score']} {user_profile['Intent']}"


        # Sentiment analysis
        sentiment = sentiment_analyzer(user_text)[0]
        #st.write(f"**Sentiment Analysis:** {sentiment['label']} (Confidence: {sentiment['score']:.2f})")

        # Generate user embedding
        user_embedding = embedding_model.encode([user_text])
        user_embedding = np.array(user_embedding, dtype=np.float32)

        # Find top 3 recommended categories
        product_data = customers["Category"].dropna().unique().tolist()
        product_embeddings = embedding_model.encode(product_data)
        product_embeddings = np.array(product_embeddings, dtype=np.float32)

        product_index = faiss.IndexFlatL2(product_embeddings.shape[1])
        product_index.add(product_embeddings)

        D, I = product_index.search(user_embedding, k=2)  # Get top 2 matches
        recommendations = [product_data[i] for i in I[0]]

        # Display recommendations
        st.write("## Recommended for You")
        for rec in recommendations:
            st.write(f"- {rec}")
        return 1;

# Prompt-based Suggestion Input
st.subheader("### Get Suggestions Based on a Prompt Message")
prompt_message = st.text_input("Enter a prompt message for suggestions:").strip()



if customer_id:
    st.empty()
    isfound = getCustomerData(customer_id)
    if isfound != 1:
        st.warning("Please enter a valid Customer ID.")




def generate_prompt_based_suggestion(prompt_message):
    st.subheader("## Generating Suggestion Based on Prompt...")

    # Concatenate Age, Gender, Interests, Preferences into a comprehensive profile text
    customer_profiles = customers[["Age", "gender", "Interests", "Preferences", "Category"]].fillna("").astype(str)
    profile_texts = customer_profiles.apply(lambda row: " ".join(row), axis=1).tolist()

    # Encode prompt and customer profiles
    prompt_embedding = embedding_model.encode([prompt_message])
    profile_embeddings = embedding_model.encode(profile_texts)

    # Convert to FAISS-compatible format
    prompt_embedding = np.array(prompt_embedding, dtype=np.float32)
    profile_embeddings = np.array(profile_embeddings, dtype=np.float32)

    # Use FAISS to find the closest matching customer profiles
    faiss_index = faiss.IndexFlatL2(profile_embeddings.shape[1])
    faiss_index.add(profile_embeddings)
    distances, indices = faiss_index.search(prompt_embedding, k=3)

    # Get top 3 matching profiles
    matching_profiles = customers.iloc[indices[0]]

    st.subheader("## Suggested Recommendations:")
    for i, profile in matching_profiles.iterrows():
        suggestion = f"Based on recent purchases we would suggest you the following " \
                     f"Interests: {profile['Interests']}, Preferences: {profile['Preferences']}, Category: {profile['Category']}), "
        st.write(f"- {suggestion}")
        break

if prompt_message:

    st.empty()
    if getCustomerData(prompt_message) != 1:
        generate_prompt_based_suggestion(prompt_message)




