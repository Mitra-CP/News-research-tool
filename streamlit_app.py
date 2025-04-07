# (newfintech) C:\Users\mitra\OneDrive\Documents\selfstudy\newfintech>streamlit run streamlit_app.py


import os
import streamlit as st
import pickle
import time
import langchain
import faiss
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

# from langchain import OpenAI
# from langchain.document_loaders import UnstructuredURLLoader
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import FAISS

#to ensure compatibility with newer versions, updated my imports
from langchain_community.llms import OpenAI
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key
#print("Loaded key:", os.getenv("OPENAI_API_KEY"))


st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls=[]
for i in range(3):
    url = st.sidebar.text_input(f"URL{i+1}")
    urls.append(url)

process_URL_clicked=st.sidebar.button("Process URLs")
file_path = './faiss_index_new1/index.pkl' #placed here after running the code ones

main_placefolder = st.empty()
llm = OpenAI(temperature=0.9,max_tokens=500)
st.write("----------------------------------------------------------------------------------------------------------------------------")
if process_URL_clicked:
    #loading data
    loader = UnstructuredURLLoader(urls=urls)
    main_placefolder.text("Data loading started .........✅✅✅")
    data = loader.load()

    #splitting data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n','\n','.',','],
        chunk_size=1000
    )
    main_placefolder.text("Text splitter started .......✅✅✅")
    docs=text_splitter.split_documents(data)

    #create embeddings and save it to FAISS index
    embeddings = OpenAIEmbeddings()
    vectorindex_openai=FAISS.from_documents(docs,embeddings)
    main_placefolder.text("Embedding vector started building ..........✅✅✅")
    time.sleep(2)

    #save FAISS index as a pickle file
    # retriever = vectorindex_openai.as_retriever()
    st.write("------Please wait for few seconds-----")
    vectorindex_openai.save_local("faiss_index_new1")

    st.write("----Couple more----")
    # store vector index and create in local as folder with pickle and faiss file
    #do the abv in the first run later comment the filepath below and place the file path alone in beginning of code
    # file_path = 'C:/Users/mitra/OneDrive/Documents/selfstudy/OPENAI in Fintech/faiss_index_new/index.pkl'

query = main_placefolder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        # with open(file_path, "rb") as f:
            # vectorstore = pickle.load(f) #store
        vectorstore = FAISS.load_local("faiss_index_new1", OpenAIEmbeddings(),allow_dangerous_deserialization=True)
        #st.write("--------VECTOR_STORE------------")
        #st.write(vectorstore)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=vectorstore.as_retriever())  # retriever here is how we will retrieve the vector database
        result = chain({"question": query}, return_only_outputs=True) #{'answer': ' ... \n','sources': '...html'}
        st.header("Answer")
        st.subheader(result["answer"])

        #Display sources if available
        sources=result.get("sources","")
        if sources:
            st.subheader("Sources:")
            sources_list=sources.split("\n") #split the sources by newline
            for source in sources_list:
                st.write(source)






