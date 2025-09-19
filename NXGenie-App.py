import os
import io
import time
import json
import datetime
from typing import List, Dict, Any, Optional

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# -------------------- Config --------------------
st.set_page_config(page_title="NXGENIE", layout="wide")
st.title("NXGENIE")
st.markdown("Your AI Assistant for Siemens NXOpen")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# -------------------- Helpers --------------------
def list_sessions() -> list[str]:
    if not os.path.exists(DATA_DIR):
        return []
    files = [f[:-5] for f in os.listdir(DATA_DIR) if f.endswith(".json")]
    return sorted(files)

def session_path(name: str) -> str:
    safe = "".join(ch for ch in name if ch.isalnum() or ch in ("-", "_", " ", "."))
    return os.path.join(DATA_DIR, f"{safe}.json")

def save_session(name: str, state: dict) -> None:
    payload = {
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "persistent_context": state.get("persistent_context", ""),
        "messages": state.get("messages", []),
        "nx_version": state.get("nx_version", ""),
        "uploaded_files_meta": state.get("uploaded_files_meta", []),
    }
    with open(session_path(name), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def load_session(name: str) -> Optional[dict]:
    path = session_path(name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# -------------------- Initial Status --------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "persistent_context" not in st.session_state:
    st.session_state["persistent_context"] = ""
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = {}   # {filename: code_text}
if "uploaded_files_meta" not in st.session_state:
    st.session_state["uploaded_files_meta"] = []  # [{"name":..., "chars":...}]
if "nx_version" not in st.session_state:
    st.session_state["nx_version"] = "NX 2212"
if "current_session_name" not in st.session_state:
    st.session_state["current_session_name"] = "sesion-nx"

# -------------------- KEY --------------------
load_dotenv()
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "")

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("OpenAI API Key", value=DEFAULT_API_KEY, type="password")

    model = st.selectbox(
        "AI Model",
        options=[
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
        ],
        index=0,
    )
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.slider("Tokens limits", 256, 8192, 1600, 64)

    st.markdown("---")
    st.subheader("NX version")
    nx_version = st.selectbox(
        "Target NX version",
        options=[
            "NX 1847","NX 1872","NX 1899","NX 1953","NX 1980","NX 1988",
            "NX 2007","NX 2206","NX 2212","NX 2306","NX 2312","NX 2406"
        ],
        index=8  # NX 2212 default
    )
    st.session_state["nx_version"] = nx_version

    st.markdown("---")
    st.caption("💡 Instructions (politics/style).")
    st.session_state["persistent_context"] = st.text_area(
        "Persistent context",
        value=st.session_state.get("persistent_context", ""),
        height=160,
        placeholder=(
            "Examples:\n"
            "- Do not modify the main() function.\n"
            "- Keep variable names in English.\n"
            "- Code targeted at NX 2212.\n"
            "- Avoid external dependencies.\n"
        ),
    )

    st.markdown("---")
    st.subheader("📦 Code Files")
    uploads = st.file_uploader(
        "Upload one or more files (.cs, .py, .txt)",
        type=["cs", "py", "txt"],
        accept_multiple_files=True,
        help="You can upload multiple files to process them together."
    )
    if uploads:
        st.session_state["uploaded_files"] = {}
        st.session_state["uploaded_files_meta"] = []
        for up in uploads:
            try:
                text = up.read().decode("utf-8", errors="ignore")
            except Exception:
                text = up.getvalue().decode("utf-8", errors="ignore")
            st.session_state["uploaded_files"][up.name] = text
            st.session_state["uploaded_files_meta"].append({"name": up.name, "chars": len(text)})
        st.success(f"{len(uploads)} archivo(s) cargado(s).")

    st.markdown("---")
    st.subheader("💾 Memory (save/load session)")
    st.session_state["current_session_name"] = st.text_input("Session name", value=st.session_state.get("current_session_name","session-nx"))
    colA, colB = st.columns(2)
    with colA:
        if st.button("Save session"):
            save_session(st.session_state["current_session_name"], dict(st.session_state))
            st.success("Session saved to /data.")
    with colB:
        existing = list_sessions()
        sel = st.selectbox("Load existing session", options=["(select)"] + existing, index=0)
        if st.button("Load"):
            if sel and sel != "(select)":
                loaded = load_session(sel)
                if loaded:
                    st.session_state["persistent_context"] = loaded.get("persistent_context","")
                    st.session_state["messages"] = loaded.get("messages",[])
                    st.session_state["nx_version"] = loaded.get("nx_version","NX 2212")
                    st.session_state["uploaded_files_meta"] = loaded.get("uploaded_files_meta", [])
                    st.session_state["current_session_name"] = sel
                    st.success(f"Session '{sel}' loaded. (Files must be re-uploaded if required)")
                else:
                    st.error("The session could not be loaded.")

# -------------------- Client OpenAI --------------------
def ensure_client(key: str) -> OpenAI:
    if not key:
        st.error("🔑 API Key missing. Place it in the sidebar or in your .env file (OPENAI_API_KEY).")
        st.stop()
    return OpenAI(api_key=key)

# -------------------- Prompting --------------------
def build_system_prompt() -> str:
    ctx = st.session_state.get("persistent_context","").strip()
    nx_ver = st.session_state.get("nx_version","").strip()
    doc_url = "https://docs.sw.siemens.com/en-US/doc/209349590/PL20221117716122093.nxopen_python_ref"

    base = (
        "You are an expert assistant in NXOpen (Python) for Siemens NX.\n"
        "Always use and respect the official NXOpen Python documentation when reasoning:\n"
        f"{doc_url}\n\n"
        "Act as a reviewer and improver of journals/code: clean, documented, robust, and ready for integration.\n"
        f"Compatibility target: **{nx_ver}**. Avoid using APIs or namespaces incompatible with that version.\n\n"
        "remove any user movements inside of the code that got recorded in the journal.\n"
    )
    if ctx:
        base += f"PERMANENT INSTRUCTIONS TO FOLLOW:\n{ctx}\n\n"
    base += (
        "Recommended response format:\n"
        "1) Brief summary of what the code does.\n"
        "2) List of issues/improvements.\n"
        "3) Proposed code (a single block with the complete code, including imports/using if applicable).\n"
        "4) Integration notes for NX.\n"
    )
    return base

def call_openai(client: OpenAI, msgs: List[Dict[str,str]]) -> str:
    last_err = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=msgs,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_err = e
            time.sleep(1.2)
    raise RuntimeError(f"Contact OpenAI: {last_err}")

def render_chat():
    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

def extract_code_block(text: str) -> str | None:
    import re
    m = re.search(r"```[a-zA-Z0-9_+-]*\n(.*?)```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return None

# -------------------- principal UI --------------------
col_actions = st.container()

with col_actions:
    st.subheader("🧹 Process code (multi-file)")
    files = st.session_state.get("uploaded_files", {})
    if files:
        st.write("**Uploaded files:**")
        for meta in st.session_state["uploaded_files_meta"]:
            st.write(f"- {meta['name']} ({meta['chars']} chars)")

        join_mode = st.radio(
            "Processing mode",
            options=["Process TOGETHER (unified)", "Process SELECTED file"],
            index=0,
        )

        if join_mode == "Process SELECTED file":
            selected = st.selectbox("Select a file to process", options=list(files.keys()))
        else:
            selected = None

        if st.button("🚀 Clean & Improve"):
            client = ensure_client(api_key)

            if join_mode == "Process SELECTED file" and selected:
                code_text = files[selected]
                code_msg = f"Received code (file: {selected}):\n```auto\n{code_text}\n```"
            else:
                # separadores
                parts = []
                for fname, code in files.items():
                    parts.append(f"// ==== ARCHIVO: {fname} ====\n{code}")
                unified = "\n\n".join(parts)
                code_msg = "Received code (multiple files unified):\n```auto\n" + unified + "\n```"

            msgs = [{"role": "system", "content": build_system_prompt()}]
            # historial de chat
            for m in st.session_state["messages"]:
                if m["role"] in ("user","assistant"):
                    msgs.append(m)
            # contenido de código
            msgs.append({"role": "user", "content": code_msg})
            msgs.append({"role": "user", "content": "Process the file(s) respecting the persistent context and the indicated NX version. Return the final code in a single block."})

            with st.spinner("Processing code..."):
                try:
                    result = call_openai(client, msgs)
                    st.success("Done! Check the result below.")
                    st.markdown("### 📄 Result")
                    st.markdown(result)

                    extracted = extract_code_block(result)
                    if extracted:
                        out_name = "improved_code.py"
                        if selected and selected.lower().endswith(".py"):
                            out_name = "improved_code.py"
                        st.download_button(
                            label="⬇️ Download proposed code",
                            data=extracted.encode("utf-8"),
                            file_name=out_name,
                            mime="text/plain",
                        )
                    else:
                        st.info("No code block formatted with ```. Copy and paste manually.")
                except Exception as e:
                    st.error(f"Ops! An error occurred: {e}")
    else:
        st.info("Upload one or more files in the sidebar to enable processing.")

st.markdown("---")
st.caption("Made by Argentina Diaz")
