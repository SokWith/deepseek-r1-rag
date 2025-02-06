import streamlit as st
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# 페이지 설정
st.set_page_config(
    page_title="Deepseek 챗봇",
    page_icon="🤖",
)

# 기본 스타일 설정
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .model-name {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        text-align: center;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# 모델 정보 표시
st.markdown('<div class="model-name">🤖 Model: Deepseek-r1:32b</div>', unsafe_allow_html=True)

# 제목
st.title("🤖 Deepseek 챗봇")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기본 프롬프트 설정
SYSTEM_PROMPT = """당신은 한국어로만 대답하는 AI 어시스턴트입니다. 
어떤 질문이 들어와도 반드시 한국어로만 답변해야 합니다.
영어나 다른 언어로 된 질문이 들어와도 한국어로 답변하세요.
전문적인 내용도 모두 한국어로 설명하세요.

규칙:
1. 항상 한국어로만 답변할 것
2. 전문 용어도 가능한 한국어로 설명할 것
3. 명확하고 이해하기 쉽게 설명할 것
4. 공손하고 정중한 말투를 사용할 것

이제 대화를 시작하겠습니다."""

# 사이드바 설정
with st.sidebar:
    st.header("설정")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    
    if st.button("대화 내역 지우기"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 사용 방법")
    st.markdown("""
    1. 메시지를 입력하세요
    2. Temperature로 창의성을 조절하세요
    3. '대화 내역 지우기'로 새로운 대화를 시작하세요
    """)

def get_llm():
    try:
        return Ollama(
            model="deepseek-r1:32b",
            temperature=temperature,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            additional_kwargs={
                "gpu": True,  # 항상 GPU 사용
                "threads": 8,
                "mmap": True
            }
        )
    except Exception as e:
        st.error(f"Ollama 연결 오류: {str(e)}")
        st.info("Ollama가 실행 중인지 확인해주세요.")
        return None
    
    
# 메시지 이력 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 채팅 입력 및 응답
if prompt := st.chat_input("메시지를 입력하세요..."):
    # 사용자 메시지 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # AI 응답 생성
    with st.chat_message("assistant"):
        try:
            llm = get_llm()
            if llm:
                message_placeholder = st.empty()
                with st.spinner("응답 생성 중..."):
                    # 시스템 프롬프트와 사용자 프롬프트 결합
                    full_prompt = f"{SYSTEM_PROMPT}\n\n사용자: {prompt}\n\n응답:"
                    
                    response = llm(full_prompt)
                    message_placeholder.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"오류가 발생했습니다: {str(e)}")
            st.info("""
            다음 사항들을 확인해주세요:
            1. Ollama가 실행 중인가요?
            2. Deepseek 모델이 설치되어 있나요? (`ollama pull deepseek` 실행)
            3. 네트워크 연결이 정상인가요?
            """)