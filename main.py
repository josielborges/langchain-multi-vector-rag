import os
import uuid

from dotenv import load_dotenv
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

from database import COLLECTION_NAME, CONNECTION_STRING
from utils.store import PostgresByteStore

load_dotenv()

OPEN_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o-mini"
id_key = "doc_id"

embeddings = OpenAIEmbeddings()
vectorstore = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=CONNECTION_STRING,
    use_jsonb=True,
)

store = PostgresByteStore(CONNECTION_STRING, COLLECTION_NAME)

retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)


def save_chuks(file_path):
    file_name = os.path.basename(file_path)

    loader = PyPDFLoader(file_path=file_path)

    # by default, we will split by pages with no text_splitter
    documents = loader.load_and_split(text_splitter=None)

    print()
    print("Chunks")
    for doc in documents:
        print(doc.model_dump())

    doc_ids = [str(uuid.uuid4()) for _ in documents]
    file_names = [file_name for _ in documents]

    print()
    print(doc_ids)

    child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

    all_sub_docs = []
    for i, doc in enumerate(documents):
        doc_id = doc_ids[i]
        sub_docs = child_text_splitter.split_documents([doc])
        for sub_doc in sub_docs:
            sub_doc.metadata[id_key] = doc_id
        all_sub_docs.extend(sub_docs)

    print()
    print("Subchunks")
    for doc in all_sub_docs:
        print(doc.model_dump())

    retriever.vectorstore.add_documents(all_sub_docs)
    retriever.docstore.mset(list(zip(doc_ids, documents, file_names)))

    return doc_ids, file_names


def test_query_chunks(text):
    child_chunks = retriever.vectorstore.similarity_search(text)
    parent_chunks = retriever.invoke(text)

    print("Child chunks")
    for chunk in child_chunks:
        print(chunk.model_dump())

    print()
    print("Parent chunks")
    for chunk in parent_chunks:
        print(chunk.model_dump())


def summarize_chunks(file_path, doc_ids, file_names):
    file_name = os.path.basename(file_path)

    loader = PyPDFLoader(file_path=file_path)

    # by default, we will split by pages with no text_splitter
    documents = loader.load_and_split(text_splitter=None)

    prompt_text = """You are an assistant tasked with summarizing text. \
    Directly summarize the following text chunk: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Initialize the Language Model (LLM)
    model = ChatOpenAI(temperature=0, model=MODEL)

    # Define the summary chain
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    parent_chunk = [i.page_content for i in documents]
    text_summaries = summarize_chain.batch(parent_chunk, {"max_concurrency": 5})

    print("Summaries")
    for summary in text_summaries:
        print(summary)

    # linking summaries to parent documents
    summary_docs = []
    for i, (summary, doc_id) in enumerate(zip(text_summaries, doc_ids)):
        # Define your new metadata here
        new_metadata = {"page": i, "doc_id": doc_id}

        # Create a new Document instance for each summary
        doc = Document(page_content=str(summary))

        # Replace the metadata
        doc.metadata = new_metadata

        # Add the Document to the list
        summary_docs.append(doc)

    print()
    print("Summary docs")
    for doc in summary_docs:
        print(doc.model_dump())

    # adding to vectorstore and docstore
    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, documents, file_names)))


def generate_hypothetical_questions(file_path, doc_ids, file_names):
    file_name = os.path.basename(file_path)

    loader = PyPDFLoader(file_path=file_path)

    # by default, we will split by pages with no text_splitter
    documents = loader.load_and_split(text_splitter=None)

    functions = [
        {
            "name": "hypothetical_questions",
            "description": "Generate hypothetical questions",
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["questions"],
            },
        }
    ]

    question_chain = (
            {"doc": lambda x: x.page_content}
            # Only asking for 5 hypothetical questions, but this could be adjusted
            | ChatPromptTemplate.from_template(
        """Generate a list of exactly 5 hypothetical questions that the below document could be used to answer:\n\n{doc}
        seperate each question with a comma (,)
        """
    )
            | ChatOpenAI(max_retries=0, model=MODEL).bind(
        functions=functions, function_call={"name": "hypothetical_questions"}
    )
            | JsonKeyOutputFunctionsParser(key_name="questions")
    )

    hypothetical_questions = question_chain.batch(documents, {"max_concurrency": 5})

    print("Hypothetical questions")
    for hypothetical_question in hypothetical_questions:
        print(hypothetical_question)

    # linking hypothetical questions to parent documents
    hypothetical_docs = []
    for question_list, doc_id in zip(hypothetical_questions, doc_ids):
        for question in question_list:
            # Define your new metadata here
            new_metadata = {"doc_id": doc_id}

            # Create a new Document instance for each question
            # The question itself is the page_content
            doc = Document(page_content=question, metadata=new_metadata)

            # Add the Document to the list
            hypothetical_docs.append(doc)

    # adding to vectorstore and docstore
    retriever.vectorstore.add_documents(hypothetical_docs)
    retriever.docstore.mset(list(zip(doc_ids, documents, file_names)))


def query_chunks(text):
    template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # LLM
    model = ChatOpenAI(temperature=1, model=MODEL)

    # RAG pipeline
    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
    )

    response = chain.invoke(text)

    print(response)


doc_ids, file_names = save_chuks("data/montreal.pdf")
summarize_chunks("data/montreal.pdf", doc_ids, file_names)
generate_hypothetical_questions("data/montreal.pdf", doc_ids, file_names)

# test_query_chunks("What are some unique seasonal events in Montreal that a visitor should not miss?")

query_chunks("What dining options are available in Montreal for those interested in Middle Eastern cuisine?")
print()
query_chunks("Where can I find the best smoked meat sandwiches in Montreal?")
print()
query_chunks("Where can I find the best food in Montreal?")
