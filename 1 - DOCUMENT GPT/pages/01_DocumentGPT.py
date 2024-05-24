import os
import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler

OPENAI_API_KEY = st.secrets["openai"]["OPENAI_API_KEY"]

# 페이지 세팅
st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

# 스트리밍을 위한 클래스
class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

# 모델 가져오기
llm = ChatOpenAI(
    temperature=0.1,
    openai_api_key=OPENAI_API_KEY,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)

# 같은 파일일 시 재로딩 방지하기
@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    # 파일 저장을 위한 경로 설정
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, "..", "..", ".cache", "files", file.name)
    
    # 필요한 디렉토리가 있는지 확인하고, 없다면 생성
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # 파일을 저장
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())

    # 파일이 존재하는지 확인
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # 문서 로드 -> 쪼개기 -> 임베딩 -> 벡터공간 저장 -> 검색기 넘기기 ( + 임베딩 캐시 작업)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")  # 각각의 파일을 임베딩
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)  # FAISS 사용
    retriever = vectorstore.as_retriever()
    return retriever

# 메세지 저장
def save_message(message, role):
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    st.session_state["messages"].append({"message": message, "role": role})

# 화면 메세지 표시
def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

# 지난 대화 표시기 
def paint_history():
    if "messages" in st.session_state:
        for message in st.session_state["messages"]:
            send_message(
                message["message"],
                message["role"],
                save=False,
            )

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

# context와 question를 요구하는 프롬프트 작성
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
            
            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)

st.title("📃 DocumentGPT")

# 마크다운 코드 작성
st.markdown(
    """
            
 당신의 파일에 대해 학습하고 답할 수 있는 chatbot!
"""
)

# streamlit의 파일 업로더
with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

# 파일이 업로드되면 입력창이 나타남
if file:
    retriever = embed_file(file)
    send_message("안녕하세요! 무엇을 도와드릴까요?", "ai", save=False)
    paint_history()
    message = st.chat_input("당신의 파일에 대해 질문하세요!")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            response = chain.invoke(message)
else:
    st.session_state["messages"] = []  # 새 파일은 히스토리 비우기
