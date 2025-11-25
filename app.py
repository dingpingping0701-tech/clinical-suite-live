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

# --- CSS 美化 (Updated with Scroll to Top button) ---
st.markdown("""
<style>
/* Streamlit 頁面主要樣式 */
div[data-testid="stExpander"] details summary p {
    font-size: 1.1rem;
    font-weight: 600;
    text-align: center;
    width: 100%;
}
div[data-testid="stButton"] button p {
    font-weight: bold;
}

/* 確保整個頁面可以平滑捲動 */
html {
    scroll-behavior: smooth;
}

/* Scroll to Top Button Style (Fixed Position) */
.scroll-to-top-btn {
    position: fixed;
    bottom: 20px;
    right: 20px;
    background-color: #1a1a1f; /* A darker shade for contrast */
    color: white;
    border: none;
    border-radius: 50%;
    width: 45px;
    height: 45px;
    cursor: pointer;
    text-align: center;
    line-height: 45px;
    font-size: 20px;
    z-index: 10000; /* Ensure it's on top */
    box-shadow: 0 4px 10px rgba[0, 0, 0, 0.4];
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

target_disease = st.text_input("請輸入病名/症狀", placeholder="請輸入病名 (例如: 敗血症, 副甲狀腺腫大)...", label_visibility="collapsed", key="target_input")

c1, c2, c3, c4 = st.columns(4)

# --- 核心邏輯：建立共通的「強制英文搜尋」指令 ---
def create_global_search_prompt(chinese_disease, required_action):
    # 這是所有快捷鍵的基底指令
    
    # 將病名翻譯成英文，確保搜尋品質
    # 注意：這裡的 Prompt 已經將「強制翻譯」寫死在 Agent 的 System Prompt 裡
    
    base_prompt = (
        f"請搜尋 [{chinese_disease}] 的最新國際指引。\n"
        f"要求：{required_action}\n"
        f"回答語言：繁體中文。"
    )
    return base_prompt

# [Btn 1] 診斷標準
with c1:
    if st.button("🩺 診斷標準", use_container_width=True):
        if not target_disease:
            st.warning("請先輸入病名 👆")
        else:
            action = (
                f"請整理：1. **評分系統**：畫出表格 + MDCalc 連結。2. **確診條件**。3. **資料來源**：附上 URL。"
            )
            st.session_state.trigger = {
                "label": f"🔍 查詢 [{target_disease}] 診斷標準",
                "query": create_global_search_prompt(target_disease, action)
            }
            st.rerun()

# [Btn 2] 實驗室檢查
with c2:
    if st.button("🧪 實驗室檢查", use_container_width=True):
        if not target_disease:
            st.warning("請先輸入病名 👆")
        else:
            action = (
                f"列出建議安排的檢查項目 (Workup)。\n"
                f"整理為：1. **血液/生化檢查** (醫學名詞優先使用英文全名與縮寫，括號內附中文解釋)。2. **影像/ECG** (附 Radiopaedia/LITFL 連結)。"
            )
            st.session_state.trigger = {
                "label": f"🔬 查詢 [{target_disease}] 實驗室檢查建議",
                "query": create_global_search_prompt(target_disease, action)
            }
            st.rerun()

# [Btn 3] 治療與目標
with c3:
    if st.button("💊 治療與目標", use_container_width=True):
        if not target_disease:
            st.warning("請先輸入病名 👆")
        else:
            action = (
                f"整理出：1. **藥物治療清單**：English Generic Name、精確劑量、頻率。2. **急性期治療目標 (Goals)**：數值與時間窗。"
            )
            st.session_state.trigger = {
                "label": f"💊 查詢 [{target_disease}] 治療藥物與目標",
                "query": create_global_search_prompt(target_disease, action)
            }
            st.rerun()

# [Btn 4] 危險徵兆
with c4:
    if st.button("⚠️ 危險徵兆", use_container_width=True):
        if not target_disease:
            st.warning("請先輸入病名 👆")
        else:
            action = (
                f"列出 [{target_disease}] 的危險徵兆 (Red Flags)。"
            )
            st.session_state.trigger = {
                "label": f"⚠️ 查詢 [{target_disease}] 危險徵兆",
                "query": create_global_search_prompt(target_disease, action)
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
            prompt = (
                f"請進行腎功能劑量調整查詢。\n"
                f"藥物：**{target_drug}**。\n"
                f"適應症：**{indication_input}**。\n"
                f"病人參數：**Cr {cr} mg/dL, CrCl {crcl} ml/min**。\n\n"
                f"請搜尋權威資料 (Sanford, Lexicomp, FDA Label)，回答：\n"
                f"1. **標準劑量**。\n"
                f"2. **此病人建議劑量 (Adjusted Dose)**：針對 CrCl {crcl} 的具體建議。\n"
                f"3. 資料來源連結 (URL)。\n"
                f"請整理成表格。說明文字用繁體中文。"
            )
            st.session_state.trigger = {
                "label": f"🧮 計算 [{target_drug}] 腎功能調整劑量 (CrCl {crcl})",
                "query": prompt
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

user_input_text = st.chat_input("輸入問題...")

# --- 核心邏輯：執行與快取 ---
final_label = ""
final_query = ""
scroll_target_id = None

if "trigger" in st.session_state:
    trigger_data = st.session_state.trigger
    
    if isinstance(trigger_data, dict):
        final_label = trigger_data["label"]
        final_query = trigger_data["query"]
        
        # --- 智慧判斷：是否已經在畫面上？ (History Click Logic) ---
        # 檢查目前所有訊息，是否有一則的 content 和 user query 相同 (且 ID 相同，以確保精準度)
        existing_msg = next((m for m in st.session_state.messages if m.get("content") == final_label and m.get("role") == "user" and m.get("id") == trigger_data.get("id")), None)
        
        if existing_msg:
            # 🎯 找到了！直接滑過去，不呼叫 AI
            scroll_target_id = existing_msg["id"]
            final_query = "" # 清空 query，這樣就不會觸發下方的 AI 執行
            final_label = "" # 清空 label，避免重複顯示
        else:
            # 如果是新的 Trigger (來自按鈕)，則生成新 ID
            scroll_target_id = get_new_id()

    del st.session_state.trigger

elif user_input_text:
    final_label = user_input_text 
    final_query = user_input_text
    scroll_target_id = get_new_id() # 新問題生成新 ID

# 如果還有 final_query，代表是新問題 (或找不到舊紀錄)
if final_query:
    history_item = {"label": final_label, "query": final_query, "id": scroll_target_id}
    
    # 確保不會重複加入歷史紀錄 (只在 query 不同時才加入)
    if not st.session_state.history or st.session_state.history[-1]["query"] != final_query:
        st.session_state.history.append(history_item)

    st.session_state.messages.append({"role": "user", "content": final_label, "id": scroll_target_id})
    
    with chat_container:
        # 新問題當場也要埋樁，不然滑不到
        st.markdown(f"<div id='{scroll_target_id}'></div>", unsafe_allow_html=True)
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
                
                # --- System Prompt: 最終國際化指令 (v30.0) ---
                system_prompt = (
                    "你是專業醫師助手 Dr. AI。\n"
                    "核心指令：\n"
                    "1. **國際化搜尋**：若使用者提問中含有中文病名，你必須將其翻譯成最精確的英文醫學術語，並優先使用該英文術語搜尋**國際權威學會或指引** (ESC, AHA, GINA, AACE, KDIGO...) 的最新資料，以確保資訊品質。\n"
                    "2. **醫學名詞呈現**：在列出檢驗項目時，優先使用**英文全名與縮寫**，並在括號內附上**繁體中文解釋** (例如: 'Parathyroid Hormone (PTH) (副甲狀腺素)')。\n"
                    "3. **藥名**：用 English Generic Name。\n"
                    "4. **劑量**：必須精確 (Specific)。\n"
                    "5. **評分系統**：畫表格 + MDCalc 連結。\n"
                    "6. **資料來源**：務必附上 URL。\n"
                    "7. **最終回答語言**：**嚴格使用繁體中文**，避免任何簡體字。"
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

# --- JavaScript 執行區 (修正版：穿透 iframe，通用滑動邏輯) ---
if scroll_target_id: # 確保有目標 ID 才執行
    js = f"""
    <script>
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
        st.session_state.messages = [{"role": "assistant", "content": "我是您的臨床助手。", "id": "init_msg"}]
        st.session_state.msg_counter = 0 # 重設訊息計數器
        st.rerun()
    
    for i, item in enumerate(reversed(st.session_state.history)):
        # 使用 item["id"] 確保點擊歷史紀錄能滑動到正確位置
        if st.button(item["label"], key=f"hist_{i}"):
            # 直接設定 trigger，觸發上方邏輯，但如果已存在則不會再次執行 AI
            st.session_state.trigger = item 
            st.rerun() # 重新運行以觸發顯示和滑動

