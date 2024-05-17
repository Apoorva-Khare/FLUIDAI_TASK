
import streamlit as st
import asyncio
from langchain_community.document_loaders import PyPDFLoader
from langchain import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.prompt_template import format_document
from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

async def generate_summary(documents):
    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0, top_p=0.7)

    # Prompt for extracting data from the documents
    doc_prompt = PromptTemplate.from_template("{page_content}")

    # Prompt for querying Gemini
    llm_prompt_template = """Write a concise summary that contains information from the same for an investor looking to evaluate the company.
    As a good summarizer focus on the key elements such as future growth prospects, key changes in the business, key triggers, important information
    that might have a material effect on next year's earnings and growth of the following:
    "{text}"
    CONCISE SUMMARY:"""
    llm_prompt = PromptTemplate.from_template(llm_prompt_template)
    stuff_chain = (
        {
            "text": lambda docs: "\n\n".join(
                format_document(doc, doc_prompt) for doc in documents
            )
        }
        | llm_prompt         # Prompt for Gemini
        | llm                # Gemini function
        | StrOutputParser()  # Output parser
    )
    summary = stuff_chain.invoke(documents)
    return summary

def main():
    st.title("PDF Summary Generator")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load the PDF document
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()
        
        # Create a button to trigger text summarization
        if st.button("Summarize PDF"):
          
          # Generate the summary asynchronously
          summary_generation = asyncio.run(generate_summary(documents))
        
          # Display the summary
          st.header("Summary")
          st.write(summary_generation)

if __name__ == "__main__":
    main()



