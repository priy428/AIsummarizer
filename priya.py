import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import requests
from bs4 import BeautifulSoup
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

def extract_news(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        return text
    except Exception as e:
        return f"Error extracting news from {url}: {e}"

summarize_prompt = PromptTemplate(
    template="Summarize the following news article:\n\n{article}\n\nSummary:",
    input_variables=["article"]
)

summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt)

def summarize_news(url):
    article = extract_news(url)
    if article.startswith("Error extracting"):
        return article
    summary = summarize_chain.run(article=article)
    return summary

# ✅ Streamlit App
st.title("📰 News Article Summarizer")

url = st.text_input("🔗 Enter the URL of a news article:")

if st.button("Summarize"):
    if url:
        with st.spinner("⏳ Summarizing..."):
            result = summarize_news(url)
        st.subheader("📝 Summary:")
        st.write(result)
    else:
        st.warning("⚠️ Please enter a valid URL.")
