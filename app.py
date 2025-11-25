import streamlit as st
import streamlit.components.v1 as components
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.callbacks import StreamlitCallbackHandler

# --- 頁面設定 ---
st.set_page_config(page_title="Dr. AI Clinical Suite", layout="wide", page_icon="⚕️")
st.title("⚕️ Dr. AI: Clinical Command Center")

# --- 讀取 Keys ---
openai_api_key = st.secrets.get("OPENAI_API_KEY")
tavily_api_key = st.secrets.get("TAVILY_API_KEY")

if not openai_api_key or not tavily_api_key:
    st.error("⚠️ 缺少 API Key，請檢查 secrets.toml")
    st.stop()

# --- CSS 美化 ---
st.markdown("""
<style>
div[data-testid="stExpander"] details summary p {
    font-size: 1.1rem;
    font-weight: 600;
    text-align: center;
    width: 100%;
}
div[data-testid="stButton"] button p {
    font-weight: bold;
}
html {
    scroll-behavior: smooth;
}
</style>
""", unsafe_allow_html=True)

# --- 初始化 Session State ---
if "messages" not in st.session_state: 
    st.session_state.messages = [{"role": "assistant", "content": "我是您的臨床助手。請輸入病名開始查詢。", "id": "init_msg"}]
if "history" not in st.session_state: 
    st.session_state.history = []
if "msg_counter" not in st.session_state:
    st.session_state.msg_counter = 0

def get_new_id():
    st.session_state.msg_counter += 1
    return f"msg_{st.session_state.msg_counter}"

# ==========================================
# 📱 主畫面控制台
# ==========================================

target_disease = st.text_input("請輸入病名/症狀", placeholder="請輸入病名 (例如: 敗血症, Sepsis)...", label_visibility="collapsed", key="target_input")

c1, c2, c3, c4 = st.columns(4)

# [Btn 1] 診斷標準
with c1:
    if st.button("🩺 診斷標準", use_container_width=True):
        if not target_disease:
            st.warning("請先輸入病名 👆")
        else:
            st.session_state.trigger = {
                "label": f"🔍 查詢 [{target_disease}] 診斷標準",
                "query": (
                    f"請搜尋最新的 [{target_disease}] 診斷指引。\n"
                    f"請整理：\n"
                    f"1. **評分系統**：表格 + MDCalc 連結。\n"
                    f"2. **確診條件**。\n"
                    f"3. **資料來源**：附上 URL。\n"
                    f"回答語言：繁體中文。"
                )
            }
            st.rerun()

# [Btn 2] 實驗室檢查
with c2:
    if st.button("🧪 實驗室檢查", use_container_width=True):
        if not target_disease:
            st.warning("請先輸入病名 👆")
        else:
            st.session_state.trigger = {
                "label": f"🧪 查詢 [{target_disease}] 實驗室檢查建議",
                "query": (
                    f"請針對疑似 [{target_disease}] 的病人，列出建議安排的檢查項目 (Workup)。\n"
                    f"請整理為：\n"
                    f"1. **血液/生化檢查**：必做與鑑別項目。\n"
                    f"2. **影像檢查**：X-ray, CT, US 等 (附 Radiopaedia 連結)。\n"
                    f"回答語言：繁體中文。"
                )
            }
            st.rerun()

# [Btn 3] 治療與目標
with c3:
    if st.button("💊 治療與目標", use_container_width=True):
        if not target_disease:
            st.warning("請先輸入病名 👆")
        else:
            st.session_state.trigger = {
                "label": f"💊 查詢 [{target_disease}] 治療藥物與目標",
                "query": (
                    f"請搜尋最新的 [{target_disease}] 治療指引。\n"
                    f"整理出：\n"
                    f"1. **藥物治療**：English Generic Name、精確劑量、頻率。\n"
                    f"2. **治療目標**：數值與時間窗。\n"
                    f"回答語言：繁體中文。"
                )
            }
            st.rerun()

# [Btn 4] 危險徵兆
with c4:
    if st.button("⚠️ 危險徵兆", use_container_width=True):
        if not target_disease:
            st.warning("請先輸入病名 👆")
        else:
            st.session_state.trigger = {
                "label": f"⚠️ 查詢 [{target_disease}] 危險徵兆",
                "query": (
                    f"請列出 [{target_disease}] 的危險徵兆 (Red Flags)。\n"
                    f"文末務必附上參考來源連結。\n"
                    f"回答語言：繁體中文。"
                )
            }
            st.rerun()

# ==========================================
# 🧮 腎功能劑量調整
# ==========================================
with st.expander("🧮 腎功能劑量調整 (Calculator)", expanded=False):
    st.caption("1. 設定藥物與適應症")
    target_drug = st.text_input("指定藥物 (必填)", placeholder="例如: Meropenem")
    
    indication_input = ""
    if target_disease:
        st.info(f"📍 適應症：**{target_disease}**")
        indication_input = target_disease
    else:
        indication_input = st.text_input("適應症 (Indication)", placeholder="例如: HAP")

    st.markdown("---")
    st.caption("2. 輸入病人數據")
    
    col_calc1, col_calc2 = st.columns(2)
    with col_calc1:
        age = st.number_input("Age", 65, step=1)
        gender = st.selectbox("Sex", ["Male", "Female"])
    with col_calc2:
        wt = st.number_input("Wt(kg)", 60.0, step=1.0)
        cr = st.number_input("Cr", 1.0, step=0.1)
    
    crcl = 0
    if cr > 0:
        crcl = ((140 - age) * wt) / (72 * cr)
        if gender == "Female": crcl *= 0.85
        crcl = round(crcl, 1)
        
        if crcl < 30:
            st.error(f"⚠️ CrCl: {crcl} ml/min")
        elif crcl < 60:
            st.warning(f"⚠️ CrCl: {crcl} ml/min")
        else:
            st.success(f"✅ CrCl: {crcl} ml/min")
    
    if st.button("🚀 計算調整後劑量", use_container_width=True, type="primary"):
        if not target_drug:
            st.warning("請輸入藥物名稱！")
        elif not indication_input:
            st.warning("請輸入適應症！")
        else:
            st.session_state.trigger = {
                "label": f"🧮 計算 [{target_drug}] 腎功能調整劑量 (CrCl {crcl})",
                "query": (
                    f"請進行腎功能劑量調整查詢。\n"
                    f"藥物：**{target_drug}**。\n"
                    f"適應症：**{indication_input}**。\n"
                    f"病人參數：**Cr {cr} mg/dL, CrCl {crcl} ml/min**。\n\n"
                    f"請搜尋權威資料 (Sanford, Lexicomp)，回答：\n"
                    f"1. **標準劑量**。\n"
                    f"2. **此病人建議劑量**：針對 CrCl {crcl} 的具體建議。\n"
                    f"3. 資料來源連結 (URL)。\n"
                    f"請整理成表格。說明文字用繁體中文。"
                )
            }
            st.rerun()

st.divider()

# ==========================================
# 💬 對話與結果區
# ==========================================
chat_container = st.container(height=500, border=True)

with chat_container:
    for msg in st.session_state.messages:
        # 1. 埋樁：建立一個空的 div，id 為該訊息的 id
        if "id" in msg:
            st.markdown(f"<div id='{msg['id']}'></div>", unsafe_allow_html=True)
        # 2. 顯示訊息
        st.chat_message(msg["role"]).write(msg["content"])

user_input_text = st.chat_input("輸入其他問題...")

final_label = ""
final_query = ""
scroll_target_id = None

if "trigger" in st.session_state:
    trigger_data = st.session_state.trigger
    
    if isinstance(trigger_data, dict):
        final_label = trigger_data["label"]
        final_query = trigger_data["query"]
        
        # --- 智慧判斷：是否已經在畫面上？ ---
        # 檢查目前所有訊息，是否有一則的 content 和 user query 相同
        existing_msg = next((m for m in st.session_state.messages if m.get("content") == final_label and m.get("role") == "user"), None)
        
        if existing_msg:
            # 🎯 找到了！直接滑過去，不呼叫 AI
            scroll_target_id = existing_msg["id"]
            # 清空 query，這樣就不會觸發下方的 AI 執行
            final_query = "" 
            final_label = "" 
    
    del st.session_state.trigger

elif user_input_text:
    final_label = user_input_text 
    final_query = user_input_text

# 如果還有 final_query，代表是新問題 (或找不到舊紀錄)
if final_query:
    # 生成新 ID
    new_id = get_new_id()
    scroll_target_id = new_id # 新問題也要滑到最下面
    
    history_item = {"label": final_label, "query": final_query}
    if not st.session_state.history or st.session_state.history[-1]["query"] != final_query:
        st.session_state.history.append(history_item)

    # 儲存時帶上 ID
    st.session_state.messages.append({"role": "user", "content": final_label, "id": new_id})
    
    with chat_container:
        # 新問題當場也要埋樁，不然滑不到
        st.markdown(f"<div id='{new_id}'></div>", unsafe_allow_html=True)
        st.chat_message("user").write(final_label)
        
        with st.chat_message("assistant"):
            # 檢查是否有歷史 Response (針對已清除對話但歷史紀錄還在的情況)
            cached_history = next((h for h in st.session_state.history if h["query"] == final_query), None)
            
            if cached_history and "response" in cached_history:
                final_ans = cached_history["response"]
                st.write(final_ans)
                st.caption("⚡️ (已讀取歷史快取)")
                st.session_state.messages.append({"role": "assistant", "content": final_ans})
            else:
                st_callback = StreamlitCallbackHandler(st.container())
                llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=openai_api_key)
                tools = [TavilySearchResults(tavily_api_key=tavily_api_key, max_results=5)]
                
                system_prompt = (
                    "你是專業醫師助手 Dr. AI。\n"
                    "任務：搜尋最新醫學指引。\n"
                    "嚴格規範：\n"
                    "1. **藥名**：English Generic Name。\n"
                    "2. **劑量**：精確數值 (Specific)。\n"
                    "3. **評分系統**：畫表格 + MDCalc 連結。\n"
                    "4. **ECG/影像**：附上 LITFL/Radiopaedia 連結。\n"
                    "5. **資料來源**：務必附上 URL。\n"
                    "語言：繁體中文。"
                )
                
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("user", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ])
                
                agent = create_openai_tools_agent(llm, tools, prompt_template)
                executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
                
                try:
                    response = executor.invoke({"input": final_query}, {"callbacks": [st_callback]})
                    final_ans = response["output"]
                    st.write(final_ans)
                    
                    # 補回 Response 到歷史紀錄，供未來快取使用
                    if st.session_state.history and st.session_state.history[-1]["query"] == final_query:
                        st.session_state.history[-1]["response"] = final_ans
                        
                    st.session_state.messages.append({"role": "assistant", "content": final_ans})
                except Exception as e:
                    st.error(f"Error: {e}")

# --- JavaScript 執行區 (修正版：穿透 iframe) ---
if scroll_target_id:
    js = f"""
    <script>
        // 嘗試尋找目標元素
        function scroll_to_target() {{
            var target = window.parent.document.getElementById('{scroll_target_id}');
            if (target) {{
                target.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
            }}
        }}
        // 延遲一點點執行，確保 DOM 已經渲染完畢
        setTimeout(scroll_to_target, 100);
    </script>
    """
    components.html(js, height=0)

# --- 側邊欄：歷史紀錄 ---
with st.sidebar:
    st.header("🕒 歷史紀錄")
    if st.button("🗑️ 清除紀錄", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    
    for i, item in enumerate(reversed(st.session_state.history)):
        if st.button(item["label"], key=f"hist_{i}"):
            st.session_state.trigger = item
            st.rerun()
