import streamlit as st
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# 页面设置
st.set_page_config(
    page_title="Deepseek 聊天机器人",
    page_icon="🤖",
)

# 基本样式设置
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

# 模型信息显示
st.markdown('<div class="model-name">🤖 模型: Deepseek-r1:32b</div>', unsafe_allow_html=True)

# 标题
st.title("🤖 Deepseek 聊天机器人")

# 会话状态初始化
if "messages" not in st.session_state:
    st.session_state.messages = []

# 基本提示设置
SYSTEM_PROMPT = """您是一个只用中文回答问题的 AI 助手。
无论收到什么语言的问题，您都必须用中文回答。
即使是专业内容，也请用中文解释。

规则：
1. 始终用中文回答
2. 尽可能用中文解释专业术语
3. 解释要清晰易懂
4. 使用礼貌、恭敬的语气

现在开始对话。"""

# 侧边栏设置
with st.sidebar:
    st.header("设置")
    temperature = st.slider("温度", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    
    if st.button("清除对话记录"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 使用方法")
    st.markdown("""
    1. 输入消息
    2. 使用温度调节创造力
    3. 点击“清除对话记录”开始新对话
    """)

def get_llm():
    try:
        return Ollama(
            model="deepseek-r1:32b",
            temperature=temperature,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        )
    except Exception as e:
        st.error(f"Ollama 连接错误: {str(e)}")
        st.info("请确保 Ollama 已启动。")
        return None
    
# 显示消息历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 聊天输入及回复
if prompt := st.chat_input("请输入消息..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 生成 AI 回复
    with st.chat_message("assistant"):
        try:
            llm = get_llm()
            if llm:
                message_placeholder = st.empty()
                with st.spinner("正在生成回复..."):
                    # 结合系统提示和用户提示
                    full_prompt = f"{SYSTEM_PROMPT}\n\n用户: {prompt}\n\n回复:"
                    
                    response = llm(full_prompt)
                    message_placeholder.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"发生错误: {str(e)}")
            st.info("""
            请检查以下事项：
            1. Ollama 是否已启动？
            2. 是否已安装 Deepseek 模型？（运行 `ollama pull deepseek`）
            3. 网络连接是否正常？
            """)
