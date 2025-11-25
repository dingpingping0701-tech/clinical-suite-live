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
/* Scroll to Top Button Style (Fixed Position) */
.scroll-to-top-btn {
    position: fixed;
    bottom: 20px;
    right: 20px;
    background-color: #1a1a1f; 
    color: white;
    border: none;
    border-radius: 50%;
    width: 45px;
    height: 45px;
    cursor: pointer;
    text-align: center;
    line-height: 45px;
    font-size: 20px;
    z-index: 10000; 
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4);
    opacity: 0.7;
    transition: opacity 0.3s;
}
.scroll-to-top-btn:hover {
    opacity: 1;
}
</style>

<button class="scroll-to-top-btn" onclick="window.parent.scrollTo({top: 0, behavior: 'smooth'});">
    ▲
</button>
""", unsafe_allow_html=True)

# --- 初始化 Session State ---
if "messages" not in st.session_state: 
    st.session_state.messages = [{"role": "assistant", "content": "我是您的臨床助手。請輸入病名開始查詢。"}]
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

# 1. 病名輸入區
target_disease = st.text_input("請輸入病名/症狀", placeholder="請輸入病名 (中英文皆可, 例如: Sepsis, 敗血症)...", label_visibility="collapsed", key="target_input")

# 2. 四大快捷鍵
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
                "label": f"🔬 查詢 [{target_disease}] 完整檢查建議",
                "query": (
                    f"請針對疑似 [{target_disease}] 的病人，列出建議安排的完整檢查。\n"
                    f"分為：1. 血液/生化 2. 影像/ECG (附 Radiopaedia/LITFL 連結)。\n"
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
                    f"2. **此病人建議劑量 (Adjusted Dose)**：針對 CrCl {crcl} 的具體建議。\n"
                    f"3. 資料來源連結 (URL)。\n"
                    f"請整理成表格。說明文字用繁體中文。"
                )
            }
            st.rerun()

st.divider()

# ==========================================
# 💬 對話與結果區 (Scrollable)
# ==========================================
chat_container = st.container(height=500, border=True)

with chat_container:
    for msg in st.session_state.messages:
        if "id" in msg:
            st.markdown(f"<div id='{msg['id']}'></div>", unsafe_allow_html=True)
        st.chat_message(msg["role"]).write(msg["content"])

final_label = ""
final_query = ""
scroll_target_id = None
is_new_query = False # 新增標誌，判斷是否為新查詢

if "trigger" in st.session_state:
    trigger_data = st.session_state.trigger
    
    if isinstance(trigger_data, dict):
        final_label = trigger_data["label"]
        final_query = trigger_data["query"]
        
        existing_msg = next((m for m in st.session_state.messages if m.get("content") == final_label and m.get("role") == "user"), None)
        
        if existing_msg:
            scroll_target_id = existing_msg["id"]
            final_query = "" 
            final_label = "" 
            st.session_state.messages[-1] = existing_msg
        else:
            is_new_query = True
    
    del st.session_state.trigger

if final_query:
    new_id = get_new_id()
    scroll_target_id = new_id
    is_new_query = True # 確保是新查詢

    history_item = {"label": final_label, "query": final_query, "id": new_id}
    if not st.session_state.history or st.session_state.history[-1]["query"] != final_query:
        st.session_state.history.append(history_item)

    st.session_state.messages.append({"role": "user", "content": final_label, "id": new_id})
    
    with chat_container:
        st.markdown(f"<div id='{new_id}'></div>", unsafe_allow_html=True)
        st.chat_message("user").write(final_label)
        
        with st.chat_message("assistant"):
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
                    "核心指令：\n"
                    "1. **國際化搜尋**：若使用者提問中含有中文病名，你必須將其翻譯成最精確的英文醫學術語，並優先使用該英文術語搜尋**國際權威學會或指引** (ESC, AHA, GINA, AACE, KDIGO...) 的最新資料，以確保資訊品質。\n"
                    "2. **醫學名詞呈現**：在列出檢驗項目時，優先使用**英文全名與縮寫**，並在括號內附上**繁體中文解釋** (例如: 'Parathyroid Hormone (PTH) (副甲狀腺素)'，避免簡體字)。\n"
                    "3. **藥名**：用 English Generic Name。\n"
                    "4. **劑量**：必須精確 (Specific)。\n"
                    "5. **評分系統**：畫表格 + MDCalc 連結。\n"
                    "6. **資料來源**：務必附上 URL。\n"
                    "7. **最終回答**：使用繁體中文。"
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
                    
                    if st.session_state.history and st.session_state.history[-1]["query"] == final_query:
                        st.session_state.history[-1]["response"] = final_ans
                        
                    st.session_state.messages.append({"role": "assistant", "content": final_ans})
                except Exception as e:
                    st.error(f"Error: {e}")

# --- JavaScript 執行區 (最終滑動邏輯) ---
if scroll_target_id:
    # 判斷是新查詢還是歷史紀錄點擊
    delay_ms = 1000 if is_new_query else 100 

    js_code = f"""
    <script>
        function scroll_to_target() {{
            var target = window.parent.document.getElementById('{scroll_target_id}');
            if (target) {{
                target.scrollIntoView({{behavior: 'smooth', block: 'start'}}); 
            }}
        }}
        // 新查詢延遲 1000ms 確保 AI 內容渲染完畢
        setTimeout(scroll_to_target, {delay_ms}); 
    </script>
    """
    components.html(js_code, height=0)

# --- 側邊欄：歷史紀錄 ---
with st.sidebar:
    st.header("🕒 歷史紀錄")
    if st.button("🗑️ 清除紀錄", use_container_width=True):
        st.session_state.history = []
        st.session_state.messages = [{"role": "assistant", "content": "我是您的臨床助手。", "id": "init_msg"}]
        st.session_state.msg_counter = 0 
        st.rerun()
    
    for i, item in enumerate(reversed(st.session_state.history)):
        if st.button(item["label"], key=f"hist_{i}"):
            st.session_state.trigger = item
            st.rerun()
