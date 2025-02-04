import streamlit as st
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# 페이지 설정
st.set_page_config(
    page_title="Deepseek-r1 Chatbot",
    layout="wide"
)

# 세션 상태 초기값 설정
if "messages" not in st.session_state:
    st.session_state.messages = []
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7
if "show_settings" not in st.session_state:
    st.session_state.show_settings = False
if "show_developer" not in st.session_state:
    st.session_state.show_developer = False
if "show_model" not in st.session_state:
    st.session_state.show_model = False

# 사용자 정의 CSS
st.markdown("""
    <style>
    /* Streamlit 기본 헤더 영역의 높이와 최소 높이를 10px로 줄입니다. */
    header[data-testid="stHeader"] {
        height: 10px;
        min-height: 10px;
    }
    /* 헤더 내부 아이콘들의 패딩을 조정하여 공간을 최소화합니다. */
    header[data-testid="stHeader"] * {
        padding-top: 0;
        padding-bottom: 0;
        margin-top: 0;
        margin-bottom: 0;
    }
    /* 페이지 상단 여백 추가 (헤더 높이에 맞게 조정) */
    .block-container {
        padding-top: 50px !important;
    }

    /* 채팅 메시지 스타일 */
    .stChatMessage {
        display: flex;
        margin-bottom: 10px;
        max-width: 100%;
    }
    
    .stChatMessage.user {
        justify-content: flex-end;
    }
    
    .stChatMessage.assistant {
        justify-content: flex-start;
    }
    
    .stChatMessage .element-container {
        width: auto;
        max-width: 70%;
        padding: 10px;
        border-radius: 10px;
    }
    
    .stChatMessage.user .element-container {
        background-color: #dcf8c6;
    }
    
    .stChatMessage.assistant .element-container {
        background-color: #f1f1f1;
    }
    </style>
""", unsafe_allow_html=True)

# 고정된 헤더 생성 (여기서는 <h1> 태그 사용)
st.markdown(
    '<h1 style="position: fixed; top: 0; left: 0; width: 100%; background: #000; color: white; text-align: center; z-index: 9999; padding: 15px 0; margin: 0;">Deeptok</h1>',
    unsafe_allow_html=True
)

# 사이드바 설정: 설정, 개발자 정보, 모델 정보 패널
with st.sidebar:
    # 설정 패널 표시
    if st.session_state.show_settings:
        st.markdown("## 설정")
        st.session_state.temperature = st.slider(
            "Temperature", 
            min_value=0.0, 
            max_value=1.0, 
            value=st.session_state.temperature, 
            step=0.1
        )
        if st.button("대화 내역 지우기", key="clear_chat"):
            st.session_state.messages = []
            st.experimental_rerun()
        st.markdown("---")
        st.markdown("### 사용 방법")
        st.markdown("""
        1. 메시지를 입력하세요  
        2. Temperature로 창의성을 조절하세요  
        3. '대화 내역 지우기'로 새로운 대화를 시작하세요
        """)
    
    # 개발자 정보 패널 표시
    if st.session_state.show_developer:
        st.markdown("## 개발자 정보")
        st.markdown("**개발자:** 홍길동")
        st.markdown("**연락처:** developer@example.com")
    
    # 모델 정보 패널 표시
    if st.session_state.show_model:
        st.markdown("## 모델 정보")
        st.markdown("**모델:** deepseek-r1:1.5b")
    
    # 아이콘을 우측 하단에 배치하기 위한 여백(Spacer)
    st.markdown("<div style='height:700px'></div>", unsafe_allow_html=True)
    # 오른쪽 정렬을 위한 컬럼 레이아웃: [3, 1, 1, 1] 비율 (왼쪽 여백 + 세 아이콘)
    cols = st.columns([3, 1, 1, 1])
    if cols[1].button("⚙️", key="settings_icon"):
        st.session_state.show_settings = not st.session_state.show_settings
    if cols[2].button("👨‍💻", key="developer_icon"):
        st.session_state.show_developer = not st.session_state.show_developer
    if cols[3].button("📦", key="model_icon"):
        st.session_state.show_model = not st.session_state.show_model

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

# Ollama 모델 설정
def get_llm():
    try:
        return Ollama(
            model="deepseek-r1:1.5b",
            temperature=st.session_state.temperature,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
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
            2. Deepseek 모델이 설치되어 있나요? (ollama pull deepseek 실행)
            3. 네트워크 연결이 정상인가요?
            """)