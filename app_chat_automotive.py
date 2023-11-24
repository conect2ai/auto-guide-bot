"Code for the Streamlit app for the Automotive Chatbot"
import os
from io import BytesIO
import base64
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import Milvus
from langchain.chat_models import ChatOpenAI
from pymilvus import connections, utility
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.units import inch
from reportlab.platypus.flowables import Image
from reportlab.platypus import Spacer
from PIL import Image as PilImage

MILVUS_HOST = "standalone"
MILVUS_PORT = "19530"
COLLECTION_NAME = "app"

def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if 'pdfs_processed' not in st.session_state:
        st.session_state.pdfs_processed = False
    if 'knowledge_base' not in st.session_state:
        st.session_state.knowledge_base = None
    if 'all_messages' not in st.session_state:
        st.session_state.all_messages = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if 'brand' not in st.session_state:
        st.session_state.brand = ''
    if 'model' not in st.session_state:
        st.session_state.model = ''
    if 'year' not in st.session_state:
        st.session_state.year = ''


def on_page(canvas, doc) -> None:
    """
    Add a logo to the top of each page.

    Args:
        canvas (reportlab.pdfgen.canvas.Canvas): Canvas object that can be used to draw on the page.
        doc (reportlab.platypus.doctemplate.BaseDocTemplate): Doc object that contains information about the document.
    """
    # This function will be called for each page during the PDF creation process.
    # It receives a `canvas` object that can be used to draw on the page,
    # and a `doc` object that contains information about the document.

    # Add your image file
    img_path = './img/logo.png'
    # Load your image file with PIL
    pil_image = PilImage.open(img_path)

    # Get the original width and height of the image
    orig_width, orig_height = pil_image.size

    # Define the width you want for the image in the PDF
    img_width = 1.0 * inch

    # Calculate the height based on the original image's aspect ratio
    img_height = img_width * orig_height / orig_width

    img = Image(img_path, width=img_width, height=img_height)

    # Draw image at the top of the page
    x_position = 1.09 * inch
    img.drawOn(canvas, x_position, doc.height + 1 * inch)


def export_chat_to_pdf() -> bytes:
    """
    Export the chat history to a PDF file.

    Returns:
        bytes: PDF file
    """
    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer, pagesize=letter)

    story = []
    styles = getSampleStyleSheet()
    style = styles['BodyText']
    style.alignment = 4  # Justify text

    # Add a space after the image
    story.append(Spacer(1, 0.5*inch))  # adjust the second parameter as necessary

    # Add chat messages in pairs, separated by a Spacer
    for i in range(0, len(st.session_state.all_messages), 2):
        user_msg = st.session_state.all_messages[i]
        if i + 1 < len(st.session_state.all_messages):
            bot_msg = st.session_state.all_messages[i+1]
        else:
            bot_msg = None

        user_text = 'You: ' + user_msg['message']
        para = Paragraph(user_text, style)
        story.append(para)

        if bot_msg:
            bot_text = 'Bot: ' + bot_msg['message']
            para = Paragraph(bot_text, style)
            story.append(para)

        # Add a Spacer after each user-bot pair
        story.append(Spacer(1, 0.2*inch))

    # The function `on_page` will be called for each page
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)

    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes


def process_pdfs(pdfs) -> None:
    """
    Process PDFs and create a knowledge base.

    Args:
        pdfs (list): List of PDF files.
    """

    # Processing PDFs
    with st.spinner('Processing PDFs...'):
        for pdf in pdfs:
            text = ''

            # Read PDF and extract text
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()

            # Extract metadata from filename (brand_model_year.pdf)
            filename = pdf.name
            brand, model, year = filename.rstrip('.pdf').split('_')

            # Split text into chunks
            text_splitter = CharacterTextSplitter(
                separator='\n',
                chunk_size=500,
                chunk_overlap=20,
                length_function=len
            )

            chunks = text_splitter.split_text(text)

            # Create embeddings
            embeddings = OpenAIEmbeddings(chunk_size=500)

            # Create metadata for each chunk
            metadata = [{'brand': brand, 'model': model, 'year': year} for _ in chunks]

            if st.session_state.knowledge_base is not None:
                # If a knowledge base is provided,
                # insert the new texts and their metadata into
                # the existing knowledge base

                st.session_state.knowledge_base.add_texts(chunks, metadata)

            else:
                # If no knowledge base is provided, create a new one
                st.session_state.knowledge_base = Milvus.from_texts(chunks,
                                                                    embeddings,
                                                                    metadata,
                                                                    connection_args={"host": MILVUS_HOST,
                                                                                     "port": MILVUS_PORT},
                                                                    collection_name=COLLECTION_NAME,
                                                                    search_params = {"metric_type": "L2",
                                                                                     "params": {"nprobe": 10},
                                                                                     "offset": 5})

        # Update session state variables
        st.session_state.pdfs_processed = True
        st.success('PDFs processed. You may now ask questions.')


def answer_question(question, brand=None, model=None, year=None) -> str:
    """
    Generate an answer to a question using the knowledge base.
    
    Args:
        question (str): Question.
        brand (str): Brand of the car.
        model (str): Model of the car.
        year (str): Year of the car.
    
    Returns:
        str: Answer to the question.
    """

    # Create filter query
    filter_query = []
    if brand is not None:
        filter_query.append(f'brand == "{brand}"')
    if model is not None:
        filter_query.append(f'model == "{model}"')
    if year is not None:
        filter_query.append(f'year == "{year}"')

    # Join the filter conditions with '&&'
    filter_query = ' &&'.join(filter_query)

    # st.write(filter_query)

    with st.spinner('Thinking...'):
        # Use similarity_search instead of similarity_search_by_vector
        docs = st.session_state.knowledge_base.similarity_search(
            query=question,
            k=10,
            param=None,
            expr=filter_query
        )

        # st.write(docs)

        # QA chain using GPT-4
        llm = ChatOpenAI(model_name='gpt-4', temperature=0)
        print(llm)
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as callback:
            response_content = chain.run(input_documents=docs, question=question)
            print(callback)

    return response_content


def clear_chat_history() -> None:
    """Clear the chat history."""
    st.session_state.messages = [{"role": "assistant",
                                  "message": "How can I help you with your automotive manual inquiries today?"}]
    st.session_state.all_messages = []

def setup_sidebar() -> None:
    """Setup the sidebar."""
    # Sidebar
    with st.sidebar:
        st.title('PDF Insights')
        st.write("\n")

        # Autentication section
        st.subheader("🔑 Authentication")
        openai_key = st.text_input('Enter your OpenAI API key:', type='password', key='openai_key')

        # Verify if the key is valid
        if openai_key and openai_key.startswith('sk-'):
            os.environ['OPENAI_API_KEY'] = openai_key
            st.session_state.api_key = openai_key
            st.success('API key provided ✅')

            # Verify if the initial message was displayed
            if not st.session_state.get('initial_message_displayed', False):
                st.session_state.messages = [{"role": "assistant",
                                              "message": "How can I help you with your automotive manual inquiries today?"}]
                st.session_state.initial_message_displayed = True
        else:
            st.warning('Please enter your OpenAI API key ⚠️')

            # Stop the app if the key is invalid
            st.stop()

        st.write("\n")

        # PDF processing section
        st.subheader("📄 PDF Processing Actions")
        pdfs = st.file_uploader('Choose PDF files', type=['pdf'], accept_multiple_files=True)
        process_pdfs_button = st.button('🔄 Process PDFs')

        # Process PDFs if the button is clicked and PDFs are uploaded
        if process_pdfs_button and pdfs:
            process_pdfs(pdfs)

        st.write('\n')

def connection_milvus() -> bool:
    """
    Connect to Milvus and create a knowledge base if the collection exists.

    Returns:
        bool: True if the collection exists, False otherwise.
    """

    # Connect to Milvus
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

    # Verify if the collection exists
    collection_exists = utility.has_collection(COLLECTION_NAME)

    # Create embeddings
    embeddings = OpenAIEmbeddings(chunk_size=500)
    # print(embeddings)

    # Create a knowledge base if the collection exists
    if collection_exists:
        st.session_state.knowledge_base = Milvus(collection_name=COLLECTION_NAME,
                                                embedding_function=embeddings,
                                                connection_args={"host": MILVUS_HOST,
                                                                 "port": MILVUS_PORT})
        st.session_state.pdfs_processed = True
    else:
        st.session_state.knowledge_base = None

    return collection_exists

def display_chat(brand=None, model=None, year=None) -> None:
    """
    Display the chat.

    Args:
        brand (str): Brand of the car.
        model (str): Model of the car.
        year (str): Year of the car.
    """

    # For each message in the session state, display the message
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["message"])

    # Input field for the user for the chat
    if prompt := st.chat_input():
        # Add the user message to the session state
        st.session_state.messages.append({"role": "user", "message": prompt})
        st.session_state.all_messages.append({"role": "user", "message": prompt})

        # Generate a response to the user's question
        response = answer_question(prompt, brand=brand, model=model, year=year)

        # Add the response of the assistant to the session state
        st.session_state.messages.append({"role": "assistant", "message": response})
        st.session_state.all_messages.append({"role": "assistant", "message": response})

        # Show the user's question and the assistant's response
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            st.write(response)

def main():
    """Main function. This function is the entry point of the Streamlit app."""

    # Initialize session state variables
    initialize_session_state()

    # Setup sidebar
    setup_sidebar()

    # Connect to Milvus and create a knowledge base if the collection exists
    collection_exists = connection_milvus()

    # Condictions to display the chat, the filters section and some buttons
    if collection_exists or st.session_state.pdfs_processed:
        with st.sidebar:
            # Filters section by brand, model and year
            st.subheader("🔍 Filter by")
            brand = st.text_input('Brand', st.session_state.brand)
            model = st.text_input('Model', st.session_state.model)
            year = st.text_input('Year', st.session_state.year)
            st.write("\n")

            # Chat actions section
            st.subheader("💬 Chat Actions")
            st.button('🧹 Clear Chat History', on_click=clear_chat_history)
            export_chat_button = st.button('📥 Export Chat')

            st.write("\n")

            # Collection actions section
            st.subheader("📚 Collection Actions")

            # Drop collection if the button is clicked
            if st.button('🧹 Clear Collection'):
                utility.drop_collection(COLLECTION_NAME)
                st.markdown('<meta http-equiv="refresh" content="1">', unsafe_allow_html=True)

        # Verify if the filters are filled
        is_input = brand and model and year

        # Display the chat if the filters are filled
        if is_input:
            # Display the chat
            display_chat(brand=brand,
                         model=model,
                         year=year)

            # Export the chat to PDF if the quantity of
            # messages is greater than 0 and the button is clicked
            if len(st.session_state.all_messages) > 0:
                if export_chat_button:
                    pdf_bytes = export_chat_to_pdf()
                    b64 = base64.b64encode(pdf_bytes).decode()
                    linko= f'<a href="data:application/octet-stream;base64,{b64}" download="chat_history.pdf">Click Here to download your PDF file</a>'
                    st.markdown(linko, unsafe_allow_html=True)
        else:
            st.warning('⚠️ Please enter the filters to activate the chat feature.')

if __name__ == '__main__':
    main()
