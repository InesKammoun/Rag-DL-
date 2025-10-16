import streamlit as st
import requests

st.set_page_config(page_title="💸 FinTech RAG Chatbot", layout="centered")
st.title("💸 FinTech RAG Chatbot")
st.markdown("""
Ask your questions about **finance, banking, cryptocurrency, or FinTech**.  
The answer will be generated using your RAG API.
""")

# Sidebar settings
st.sidebar.header("⚙️ Settings")
top_k = st.sidebar.slider("Top K search results", 1, 10, 5)
final_k = st.sidebar.slider("Final K chunks for answer", 1, 5, 3)
window_size = st.sidebar.slider("Window size (0=disabled)", 0, 3, 1)

# User query input
query = st.text_input("💬 Type your FinTech question:")

if st.button("Ask") and query:
    with st.spinner("🤖 Thinking..."):
        try:
            response = requests.post(
                "http://localhost:8000/answer",
                json={
                    "query": query,
                    "top_k": top_k,
                    "final_k": final_k,
                    "window_size": window_size
                },
                timeout=60
            )

            # Si c’est du texte brut (PlainTextResponse)
            if "text" in response.headers.get("content-type", ""):
                answer_text = response.text.strip()
            else:
                answer_text = response.json().get("answer", "No answer returned.")

            # 🔹 Affichage clair et lisible
            st.markdown(f"### 🧠 Answer:\n\n{answer_text}")

        except requests.exceptions.RequestException as e:
            st.error(f"❌ API request failed: {e}")

# ================================
# 🔍 Search raw passages
# ================================
st.markdown("---")
st.subheader("🔍 Search in FinTech Documents")

raw_query = st.text_input("Enter a keyword:", key="raw")

if st.button("Search", key="search_raw") and raw_query:
    with st.spinner("Searching relevant passages..."):
        try:
            response = requests.post(
                "http://localhost:8000/search",
                json={
                    "query": raw_query,
                    "top_k": top_k,
                    "final_k": final_k,
                    "window_size": window_size
                },
                timeout=60
            )

            if "application/json" in response.headers.get("content-type", ""):
                results = response.json().get("results", [])
                if results:
                    for i, r in enumerate(results, 1):
                        # ✅ Utilisation du score positif
                        score_value = r.get("score", 0)
                        st.markdown(f"**📄 Passage {i}** *(score: {score_value:.4f})*")
                        st.write(r["text"])
                else:
                    st.info("No matching passages found.")
            else:
                st.write(response.text)

        except requests.exceptions.RequestException as e:
            st.error(f"❌ API request failed: {e}")
