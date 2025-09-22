
import os
import time
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.embeddings import CohereEmbeddings
# import google.generativeai as genai
from langchain.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# genai.configure(api_key = os.getenv('GOOGLE_API_KEY'))



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text




def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        length_function = len,
        separator='\n',
        chunk_size = 1000,
        chunk_overlap = 200
    )

    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(chunks):

    embeddings = CohereEmbeddings(cohere_api_key= os.getenv("COHERE_API_KEY"),
        model="embed-english-light-v3.0",
        user_agent="langchain")

    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local('faiss_index')
    return vector_store



# def get_vectorstore(text_chunks, batch_size=100):
#     """Embeds text chunks in batches with a delay to avoid rate limiting."""
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = None
    
#     for i in range(0, len(text_chunks), batch_size):
#         batch = text_chunks[i:i + batch_size]
        
#         if vector_store is None:
#             # Create the store with the first batch
#             vector_store = FAISS.from_texts(batch, embedding=embeddings)
#         else:
#             # Add subsequent batches
#             vector_store.add_texts(batch)
            
#         print(f"Processed batch {i // batch_size + 1}, waiting 60 seconds...")
#         # The free tier limit is often per minute, so a 60-second wait is safe.
#         time.sleep(65) 
        
#     return vector_store

# In your main() function, replace the old call with this one:
# vectore_db = get_vectorstore(chunks)  <-- Your old code
# vectore_db = get_vectorstore(chunks) # <-- Your new code

def get_conversational_chain(vector_store):
    my_prompt_template = """"
    Answer the questions as detailed as possible from the probided context, make sure to provide all details, if the
    answer is not in the provided context just say , answer is not available in the context, don't provide the wrong answer\n
    Context: \n{context}?\n
    Question: \n{question}\n
    No preamble
    Answer:
    """

    model = ChatGroq(temperature=.34, groq_api_key = os.getenv("GROQ_API_KEY"), model='llama-3.3-70b-versatile')

    prompt = PromptTemplate(template = my_prompt_template, input_variables = ["context","question"])

    chain = load_qa_chain(model, chain_type='stuff', prompt = prompt)

    return chain



def user_query(question):
    embeddings = CohereEmbeddings(cohere_api_key= os.getenv("COHERE_API_KEY"),
        model="embed-english-light-v3.0",
        user_agent="langchain")

    vector_db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
    # vector_db = 
    docs = vector_db.similarity_search(question)

    chain = get_conversational_chain(vector_db)

    response = chain(
        {"input_documents":docs, 
         "question":question},
        return_only_outputs = True
        
    )
    print(response)
    st.write("Reply: ", response['output_text'])



def main():
    load_dotenv()
    st.set_page_config(page_title = "Chat with multiple PDF's", page_icon = ":books:")

    st.header("Chat with multiple PDF's :books:")
    user_question = st.text_input("Ask the question about your PDF's")
    if user_question:
        user_query(user_question)



    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your Pdf's here and click on 'process'", accept_multiple_files=True)
        if st.button("process"):
            with st.spinner('Processing'):
                # get the pdf documents
                raw_text = get_pdf_text(pdf_docs)
                
                # get the text chunks
                chunks = get_text_chunks(raw_text)
                # st.write(chunks)
                
                # create the vetorstore
                vectore_db = get_vectorstore(chunks)






if __name__ == "__main__":
    main()