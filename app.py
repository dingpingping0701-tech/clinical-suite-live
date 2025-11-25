import streamlit as st
import streamlit.components.v1 as components
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.callbacks import StreamlitCallbackHandler

# --- 頁面設定 ---
st.set_page_config(page_title="Dr. AI Clinical Suite", layout="wide", page_icon="⚕️")

# --- 埋入頂部錨點 ---
st.markdown("<div id='top_anchor'></div>", unsafe_allow_html=True)

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
.float-btn {
    position: fixed;
    bottom: 20px;
    right: 20px;
    width: 50px;
    height: 50px;
    background-color: #FF4B4B;
    color: white;
    border-radius: 50%;
    text-align: center;
    line-height: 50px;
    font-size: 24px;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
    z-index: 9999;
    text-decoration: none;
    transition: opacity 0.3s;
    opacity: 0.8;
}
.float-btn:hover {
    opacity: 1;
    color: white;
}
</style>
<a href="#top_anchor" class="float-btn" target="_self">▲</a>
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

# 1. 病名輸入 (主畫面)
target_disease = st.text_input("請輸入病名/症狀", placeholder="請輸入病名 (中英文皆可, 例如: Sepsis)...", label_visibility="collapsed", key="target_input")

# 2. 四大快捷鍵 (主畫面)
c1, c2, c3, c4 = st.columns(4)

def handle_button_click(label_template, query_template):
    if not target_disease:
        st.warning("請先輸入病名 👆")
    else:
        st.session_state.trigger_action = {
            "type": "new_search",
            "label": label_template.format(target_disease),
            "query": query_template.format(target_disease)
        }
        st.rerun()

with c1:
    if st.button("🩺 診斷標準", use_container_width=True):
        q = "請搜尋最新的 [{}] 診斷指引。\n請整理：1. **評分系統**：表格 + MDCalc 連結。2. **確診條件**。3. **資料來源**：附上 URL。\n回答語言：繁體中文。"
        handle_button_click("🔍 查詢 [{}] 診斷標準", q)

with c2:
    if st.button("🧪 實驗室檢查", use_container_width=True):
        q = "請針對疑似 [{}] 的病人，列出建議安排的檢查項目 (Workup)。\n整理為：1. **血液/生化檢查** (醫學名詞優先使用英文全名與縮寫，括號內附中文解釋)。2. **影像/ECG** (附 Radiopaedia/LITFL 連結)。"
        handle_button_click("🔬 查詢 [{}] 完整檢查建議", q)

with c3:
    if st.button("💊 治療與目標", use_container_width=True):
        q = "請搜尋最新的 [{}] 治療指引。\n整理出：1. **藥物治療清單**：English Generic Name、精確劑量、頻率。2. **急性期治療目標 (Goals)**：數值與時間窗。\n回答語言：繁體中文。"
        handle_button_click("💊 查詢 [{}] 治療藥物與目標", q)

with c4:
    if st.button("⚠️ 危險徵兆", use_container_width=True):
        q = "請列出 [{}] 的危險徵兆 (Red Flags)。\n文末務必附上參考來源連結。\n回答語言：繁體中文。"
        handle_button_click("⚠️ 查詢 [{}] 危險徵兆", q)

# 3. 腎功能計算機 (主畫面)
with st.expander("🧮 腎功能劑量調整 (Calculator)", expanded=False):
    st.caption("1. 設定藥物與適應症")
    target_drug = st.text_input("指定藥物 (必填)", placeholder="例如: Meropenem")
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
        if crcl < 30: st.error(f"⚠️ CrCl: {crcl} ml/min")
        elif crcl < 60: st.warning(f"⚠️ CrCl: {crcl} ml/min")
        else: st.success(f"✅ CrCl: {crcl} ml/min")
    
    if st.button("🚀 計算調整後劑量", use_container_width=True, type="primary"):
        if not target_drug:
            st.warning("請輸入藥物名稱！")
        elif not indication_input:
            st.warning("請輸入適應症！")
        else:
            q = f"請進行腎功能劑量調整查詢。\n藥物：**{target_drug}**。\n適應症：**{indication_input}**。\n病人參數：**Cr {cr} mg/dL, CrCl {crcl} ml/min**。\n\n請搜尋權威資料，回答：1. **標準劑量**。2. **此病人建議劑量 (Adjusted Dose)**：針對 CrCl {crcl} 的具體建議。3. 資料來源連結。\n請整理成表格。說明文字用繁體中文。"
            st.session_state.trigger_action = {
                "type": "new_search",
                "label": f"🧮 計算 [{target_drug}] 腎功能調整劑量 (CrCl {crcl})",
                "query": q
            }
            st.rerun()

st.divider()

# ==========================================
# 💬 對話與結果區
# ==========================================
chat_placeholder = st.container() 

with chat_placeholder:
    for msg in st.session_state.messages:
        if "id" in msg:
            st.markdown(f"<div id='{msg['id']}'></div>", unsafe_allow_html=True)
        st.chat_message(msg["role"]).write(msg["content"])

# --- 邏輯處理核心 ---
final_label = ""
final_query = ""
scroll_target_id = None
should_run_api = False

if "trigger_action" in st.session_state:
    action = st.session_state.trigger_action
    
    if action["type"] == "history_click":
        target_id = action.get("id")
        existing_msg = next((m for m in st.session_state.messages if m.get("id") == target_id), None)
        
        if existing_msg:
            scroll_target_id = target_id
        else:
            # 恢復舊訊息
            history_item = next((h for h in st.session_state.history if h.get("id") == target_id), None)
            if history_item and "response" in history_item:
                st.session_state.messages.append({"role": "user", "content": history_item["label"], "id": target_id})
                st.session_state.messages.append({"role": "assistant", "content": history_item["response"]})
                scroll_target_id = target_id
                st.rerun()

    elif action["type"] == "new_search":
        final_label = action["label"]
        final_query = action["query"]
        should_run_api = True
    
    del st.session_state.trigger_action

# 執行 API
if should_run_api and final_query:
    new_id = get_new_id()
    scroll_target_id = new_id
    
    st.session_state.messages.append({"role": "user", "content": final_label, "id": new_id})
    
    with chat_placeholder:
        st.markdown(f"<div id='{new_id}'></div>", unsafe_allow_html=True)
        st.chat_message("user").write(final_label)
        
        with st.chat_message("assistant"):
            cached_item = next((h for h in st.session_state.history if h["query"] == final_query and "response" in h), None)
            
            if cached_item:
                final_ans = cached_item["response"]
                st.write(final_ans)
                st.caption("⚡️ (已讀取歷史快取)")
                st.session_state.messages.append({"role": "assistant", "content": final_ans})
                cached_item["id"] = new_id 
            else:
                st_callback = StreamlitCallbackHandler(st.container())
                llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=openai_api_key)
                tools = [TavilySearchResults(tavily_api_key=tavily_api_key, max_results=5)]
                
                system_prompt = (
                    "你是專業醫師助手 Dr. AI。\n"
                    "核心指令：\n"
                    "1. **國際化搜尋**：中文病名自動轉英文搜尋，回答用 **繁體中文**。\n"
                    "2. **醫學名詞**：優先用英文全名/縮寫 + 繁體中文解釋 (避免簡體)。\n"
                    "3. **藥名**：一律用 English Generic Name。\n"
                    "4. **劑量**：精確數值。\n"
                    "5. **連結**：MDCalc/Radiopaedia/LITFL + 來源 URL。\n"
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
                    
                    # API 完成後，確保歷史紀錄被正確加入 (新查詢且無重複時)
                    new_history_item = {"label": final_label, "query": final_query, "response": final_ans, "id": new_id}
                    # 檢查最後一筆是否相同，若不同才加入
                    if not st.session_state.history or st.session_state.history[-1]["query"] != final_query:
                        st.session_state.history.append(new_history_item)
                        
                    st.session_state.messages.append({"role": "assistant", "content": final_ans})
                except Exception as e:
                    st.error(f"Error: {e}")

# --- JavaScript 滑動邏輯 ---
if scroll_target_id:
    js = f"""
    <script>
        setTimeout(function() {{
            var target = window.parent.document.getElementById('{scroll_target_id}');
            if (target) {{
                target.scrollIntoView({{behavior: 'smooth', block: 'start'}}); 
            }}
        }}, 1000);
    </script>
    """
    components.html(js, height=0)

# ==========================================
# 側邊欄：歷史紀錄 (移到最下方)
# ==========================================
with st.sidebar:
    st.header("🕒 歷史紀錄")
    if st.button("🗑️ 清除紀錄", use_container_width=True):
        st.session_state.history = []
        st.session_state.messages = [{"role": "assistant", "content": "我是您的臨床助手。", "id": "init_msg"}]
        st.session_state.msg_counter = 0 
        st.rerun()
    
    if not st.session_state.history:
        st.caption("尚無搜尋紀錄")
    else:
        for i, item in enumerate(reversed(st.session_state.history)):
            if st.button(item["label"], key=f"hist_{i}"):
                st.session_state.trigger_action = {
                    "type": "history_click",
                    "id": item.get("id")
                }
                st.rerun()
