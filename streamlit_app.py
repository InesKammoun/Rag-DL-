import streamlit as st
import requests
import time
import pandas as pd
import numpy as np

# Configuration de la page
st.set_page_config(
    page_title="💸 FinTech RAG Chatbot", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS pour le style avancé avec mode dark/light COMPLET
def load_css():
    if st.session_state.get('dark_mode', True):
        # Mode Dark COMPLET
        st.markdown("""
        <style>
        /* Main App Background */
        .stApp {
            background: linear-gradient(135deg, #0e1117 0%, #1a1a2e 50%, #16213e 100%);
            color: #ffffff;
        }
        
        /* Sidebar Dark Mode */
        .css-1d391kg, .css-1cypcdb, .css-17eq0hr {
            background: linear-gradient(180deg, #1a202c 0%, #2d3748 100%) !important;
            color: #ffffff !important;
        }
        
        /* Sidebar Header */
        .css-1v3fvcr {
            color: #ffffff !important;
        }
        
        /* Sidebar Text */
        .css-1avcm0n, .css-1629p8f {
            color: #e2e8f0 !important;
        }
        
        /* Sliders Dark */
        .stSlider > div > div > div {
            background-color: #2d3748 !important;
        }
        
        /* Buttons Dark */
        .stButton > button {
            background: linear-gradient(45deg, #667eea, #764ba2) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
        }
        
        /* Text Inputs Dark */
        .stTextInput > div > div > input {
            background-color: #2d3748 !important;
            color: #ffffff !important;
            border: 1px solid #4a5568 !important;
        }
        
        /* Success/Error Messages */
        .stSuccess {
            background-color: rgba(56, 161, 105, 0.2) !important;
            color: #68d391 !important;
        }
        
        .stError {
            background-color: rgba(245, 101, 101, 0.2) !important;
            color: #fc8181 !important;
        }
        
        .main-header {
            background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
            padding: 1.5rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
            backdrop-filter: blur(4px);
            border: 1px solid rgba(255, 255, 255, 0.18);
        }
        
        .chat-container {
            background: rgba(26, 32, 46, 0.7);
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 2rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        .search-container {
            background: rgba(45, 55, 72, 0.7);
            border-radius: 20px;
            padding: 2rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        .evaluation-container {
            background: rgba(123, 31, 162, 0.7);
            border-radius: 20px;
            padding: 2rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px rgba(123, 31, 162, 0.3);
        }
        
        .chat-message {
            padding: 1.5rem;
            border-radius: 15px;
            margin-bottom: 1rem;
            display: flex;
            align-items: flex-start;
            animation: fadeIn 0.5s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .chat-message.user {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin-left: 2rem;
        }
        
        .chat-message.assistant {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            margin-right: 2rem;
        }
        
        .chat-avatar {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            margin-right: 1rem;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
        }
        
        .chat-content {
            flex: 1;
            line-height: 1.8;
            font-size: 1.1rem;
        }
        
        .search-result {
            background: rgba(26, 32, 46, 0.8);
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
            border-left: 4px solid #38a169;
            backdrop-filter: blur(5px);
        }
        
        .metric-card {
            background: rgba(26, 32, 46, 0.8);
            border-radius: 15px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(5px);
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        # Mode Light COMPLET
        st.markdown("""
        <style>
        /* Main App Background */
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            color: #2d3748;
        }
        
        /* Sidebar Light Mode */
        .css-1d391kg, .css-1cypcdb, .css-17eq0hr {
            background: linear-gradient(180deg, #ffffff 0%, #f7fafc 100%) !important;
            color: #2d3748 !important;
        }
        
        /* Sidebar Header */
        .css-1v3fvcr {
            color: #2d3748 !important;
        }
        
        /* Sidebar Text */
        .css-1avcm0n, .css-1629p8f {
            color: #4a5568 !important;
        }
        
        /* Sliders Light */
        .stSlider > div > div > div {
            background-color: #e2e8f0 !important;
        }
        
        /* Buttons Light */
        .stButton > button {
            background: linear-gradient(45deg, #4facfe, #00f2fe) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
        }
        
        /* Text Inputs Light */
        .stTextInput > div > div > input {
            background-color: #ffffff !important;
            color: #2d3748 !important;
            border: 1px solid #e2e8f0 !important;
        }
        
        /* Success/Error Messages */
        .stSuccess {
            background-color: rgba(56, 161, 105, 0.1) !important;
            color: #2f855a !important;
        }
        
        .stError {
            background-color: rgba(245, 101, 101, 0.1) !important;
            color: #e53e3e !important;
        }
        
        .main-header {
            background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
            padding: 1.5rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 8px 32px rgba(79, 172, 254, 0.37);
            backdrop-filter: blur(4px);
            border: 1px solid rgba(255, 255, 255, 0.18);
        }
        
        .chat-container {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 2rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(0, 0, 0, 0.05);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        
        .search-container {
            background: rgba(247, 250, 252, 0.9);
            border-radius: 20px;
            padding: 2rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(0, 0, 0, 0.05);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        
        .evaluation-container {
            background: rgba(243, 244, 246, 0.9);
            border-radius: 20px;
            padding: 2rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(0, 0, 0, 0.05);
            box-shadow: 0 8px 32px rgba(123, 31, 162, 0.1);
        }
        
        .chat-message {
            padding: 1.5rem;
            border-radius: 15px;
            margin-bottom: 1rem;
            display: flex;
            align-items: flex-start;
            animation: fadeIn 0.5s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .chat-message.user {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin-left: 2rem;
            color: white;
        }
        
        .chat-message.assistant {
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            margin-right: 2rem;
            color: #2d3748;
        }
        
        .chat-avatar {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            margin-right: 1rem;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            background: rgba(255, 255, 255, 0.3);
            backdrop-filter: blur(10px);
        }
        
        .chat-content {
            flex: 1;
            line-height: 1.8;
            font-size: 1.1rem;
        }
        
        .search-result {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
            border-left: 4px solid #38a169;
            backdrop-filter: blur(5px);
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }
        
        .metric-card {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 15px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            border: 1px solid rgba(0, 0, 0, 0.05);
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }
        </style>
        """, unsafe_allow_html=True)

# =============================================
# RAGas Evaluation Functions (Simple Version)
# =============================================

def simple_rag_evaluation(num_questions=5):
    """Évaluation RAG simplifiée sans dépendances externes"""
    try:
        # Get documents from API
        response = requests.post(
            "http://localhost:8000/search",
            json={"query": "fintech blockchain cryptocurrency", "top_k": num_questions * 2, "final_k": num_questions},
            timeout=60
        )
        
        if response.status_code != 200:
            return {"error": "Failed to fetch documents"}
        
        documents = response.json().get("results", [])
        
        if not documents:
            return {"error": "No documents found"}
        
        # Simple test questions based on document content
        test_questions = [
            "What is blockchain technology?",
            "How does cryptocurrency work?",
            "What are the benefits of fintech?",
            "What are the risks in digital finance?",
            "How secure are digital payments?"
        ]
        
        results = []
        for i, question in enumerate(test_questions[:num_questions]):
            # Get RAG answer
            try:
                rag_response = requests.post(
                    "http://localhost:8000/answer",
                    json={"query": question, "top_k": 5, "final_k": 3, "window_size": 1},
                    timeout=30
                )
                
                if rag_response.status_code == 200:
                    if "application/json" in rag_response.headers.get("content-type", ""):
                        answer = rag_response.json().get("answer", "No answer")
                    else:
                        answer = rag_response.text.strip()
                else:
                    answer = "Error getting answer"
                
                # Simple scoring based on answer length and presence of keywords
                score = min(len(answer) / 100, 1.0)  # Basic length-based score
                relevance = 0.8 if any(keyword in answer.lower() for keyword in ["fintech", "blockchain", "crypto", "finance"]) else 0.3
                
                results.append({
                    "question": question,
                    "answer": answer,
                    "length_score": score,
                    "relevance_score": relevance,
                    "overall_score": (score + relevance) / 2
                })
                
            except Exception as e:
                results.append({
                    "question": question,
                    "answer": f"Error: {e}",
                    "length_score": 0,
                    "relevance_score": 0,
                    "overall_score": 0
                })
        
        # Calculate averages
        avg_length = np.mean([r["length_score"] for r in results])
        avg_relevance = np.mean([r["relevance_score"] for r in results])
        avg_overall = np.mean([r["overall_score"] for r in results])
        
        return {
            "results": results,
            "metrics": {
                "Average Length Score": avg_length,
                "Average Relevance Score": avg_relevance,
                "Overall Performance": avg_overall
            },
            "num_questions": len(results),
            "num_documents": len(documents)
        }
        
    except Exception as e:
        return {"error": f"Evaluation failed: {e}"}

def get_score_interpretation(score):
    """Interprète un score"""
    if score >= 0.8:
        return "🟢 Excellent"
    elif score >= 0.6:
        return "🟡 Good"
    elif score >= 0.4:
        return "🟠 Needs Improvement"
    else:
        return "🔴 Poor"

# Initialisation des variables de session
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'search_messages' not in st.session_state:
    st.session_state.search_messages = []
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True

# Charger le CSS
load_css()

# =============================================
# SIDEBAR - Paramètres
# =============================================
st.sidebar.header("⚙️ Settings")

# Toggle Dark/Light Mode dans la sidebar
mode_icon = "🌞" if st.session_state.dark_mode else "🌙"
if st.sidebar.button(f"{mode_icon} Toggle Mode", use_container_width=True):
    st.session_state.dark_mode = not st.session_state.dark_mode
    st.rerun()

st.sidebar.markdown("---")

# Navigation entre pages
st.sidebar.subheader("📄 Navigation")
page = st.sidebar.radio(
    "Choose a page:",
    ["💬 Chat Assistant", "🔍 Document Search", "📊 RAG Evaluation"],
    key="page_selector"
)

st.sidebar.markdown("---")

# Paramètres RAG
st.sidebar.subheader("🔧 RAG Parameters")
top_k = st.sidebar.slider("🔍 Top K search results", 1, 15, 5)
final_k = st.sidebar.slider("📄 Final K chunks", 1, 8, 3)
window_size = st.sidebar.slider("🪟 Window size", 0, 5, 1)

st.sidebar.markdown("---")

# API Status Check
st.sidebar.subheader("🌐 API Status")
try:
    response = requests.get("http://localhost:8000/ping", timeout=2)
    if response.status_code == 200:
        st.sidebar.success("✅ API Connected")
        api_available = True
    else:
        st.sidebar.error("❌ API Error")
        api_available = False
except:
    st.sidebar.error("❌ API Offline")
    api_available = False

st.sidebar.markdown("---")

# Rebuild Index
st.sidebar.subheader("🔄 Index Management")
if st.sidebar.button("🔄 Rebuild Index", use_container_width=True):
    with st.sidebar.spinner("Rebuilding..."):
        try:
            response = requests.post("http://localhost:8000/rebuild", timeout=300)
            if response.status_code == 200:
                st.sidebar.success("✅ Index Rebuilt")
            else:
                st.sidebar.error("❌ Rebuild Failed")
        except:
            st.sidebar.error("❌ Rebuild Error")

# Clear History pour la page active
if page == "💬 Chat Assistant":
    if st.sidebar.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
elif page == "🔍 Document Search":
    if st.sidebar.button("🗑️ Clear Search History", use_container_width=True):
        st.session_state.search_messages = []
        st.rerun()
elif page == "📊 RAG Evaluation":
    if st.sidebar.button("🗑️ Clear Evaluation", use_container_width=True):
        if 'evaluation_results' in st.session_state:
            del st.session_state['evaluation_results']
        st.rerun()

# =============================================
# HEADER PRINCIPAL
# =============================================
st.markdown("""
<div class="main-header">
    <h1 style="margin: 0; font-size: 2.5rem; font-weight: bold;">💸 FinTech RAG Chatbot</h1>
    <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">
        Ask questions about finance, banking, cryptocurrency, or FinTech
    </p>
</div>
""", unsafe_allow_html=True)

# =============================================
# PAGE 1: CHAT ASSISTANT
# =============================================
if page == "💬 Chat Assistant":
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown("## 💬 Chat with FinTech Assistant")
    st.markdown("*Get intelligent answers about FinTech based on our document database*")

    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user">
                <div class="chat-avatar">👤</div>
                <div class="chat-content">
                    <strong>You</strong><br>
                    {message["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant">
                <div class="chat-avatar">🤖</div>
                <div class="chat-content">
                    <strong>FinTech Assistant</strong><br>
                    {message["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Chat input avec form
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        
        with col1:
            user_input = st.text_input(
                "💬 Type your FinTech question...",
                placeholder="e.g., What is blockchain technology?",
                label_visibility="collapsed"
            )
        
        with col2:
            submit_button = st.form_submit_button("Send 🚀", use_container_width=True)

    if submit_button and user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Show thinking
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown(f"""
        <div class="chat-message assistant">
            <div class="chat-avatar">🤖</div>
            <div class="chat-content">
                <strong>FinTech Assistant</strong><br>
                <em>🧠 Analyzing your question...</em>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            # API call
            response = requests.post(
                "http://localhost:8000/answer",
                json={
                    "query": user_input,
                    "top_k": top_k,
                    "final_k": final_k,
                    "window_size": window_size
                },
                timeout=60
            )
            
            if response.status_code == 200:
                if "application/json" in response.headers.get("content-type", ""):
                    result = response.json()
                    answer = result.get("answer", "No answer returned.")
                else:
                    answer = response.text.strip()
            else:
                answer = f"❌ Error: {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            answer = f"❌ Connection error: {str(e)}"
        
        # Clear thinking and add response
        thinking_placeholder.empty()
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# =============================================
# PAGE 2: DOCUMENT SEARCH
# =============================================
elif page == "🔍 Document Search":
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    st.markdown("## 🔍 Search in FinTech Documents")
    st.markdown("*Find specific passages and documents related to your keywords*")

    # Display search history
    if st.session_state.search_messages:
        st.markdown("### 📋 Search History:")
        for i, search in enumerate(st.session_state.search_messages):
            with st.expander(f"🔍 Search {i+1}: {search['query'][:50]}..."):
                st.markdown(f"**Query:** {search['query']}")
                st.markdown(f"**Results:** {len(search['results'])} passages found")
                for j, result in enumerate(search['results'][:3]):  # Show top 3
                    st.markdown(f"**Passage {j+1}:** {result['text'][:200]}...")

    st.markdown("---")

    # New search
    col1, col2 = st.columns([4, 1])

    with col1:
        search_query = st.text_input(
            "🔎 Enter search keywords:",
            placeholder="e.g., cryptocurrency risks, blockchain benefits",
            key="search_input"
        )

    with col2:
        search_button = st.button("Search 📊", use_container_width=True)

    if search_button and search_query:
        with st.spinner("🔍 Searching through FinTech documents..."):
            try:
                response = requests.post(
                    "http://localhost:8000/search",
                    json={
                        "query": search_query,
                        "top_k": top_k,
                        "final_k": final_k,
                        "window_size": window_size
                    },
                    timeout=60
                )

                if "application/json" in response.headers.get("content-type", ""):
                    results = response.json().get("results", [])
                    
                    # Save to search history
                    st.session_state.search_messages.append({
                        "query": search_query,
                        "results": results
                    })
                    
                    if results:
                        st.markdown(f"### 📋 Found {len(results)} relevant passages:")
                        
                        for i, result in enumerate(results, 1):
                            score = result.get("score", 0)
                            rerank_score = result.get("rerank_score", 0)
                            text = result.get("text", "")
                            
                            # Truncate long texts
                            display_text = text[:400] + "..." if len(text) > 400 else text
                            
                            st.markdown(f"""
                            <div class="search-result">
                                <h4>📄 Passage {i}</h4>
                                <p><strong>Search Score:</strong> {score:.4f} | <strong>Relevance Score:</strong> {rerank_score:.4f}</p>
                                <p>{display_text}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show full text option
                            if len(text) > 400:
                                with st.expander(f"📖 Show full text for passage {i}"):
                                    st.write(text)
                    else:
                        st.info("🔍 No matching documents found. Try different keywords.")
                else:
                    # Handle text response
                    st.write("📄 **Raw Response:**")
                    st.write(response.text)

            except requests.exceptions.RequestException as e:
                st.error(f"❌ Search failed: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

# =============================================
# PAGE 3: RAG EVALUATION
# =============================================
elif page == "📊 RAG Evaluation":
    st.markdown('<div class="evaluation-container">', unsafe_allow_html=True)
    st.markdown("## 📊 RAG System Evaluation")
    st.markdown("*Comprehensive evaluation of your RAG system performance*")

    if not api_available:
        st.error("❌ API not available. Please start your FastAPI server first.")
    else:
        # Evaluation parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            num_questions = st.slider("Number of test questions", 3, 10, 5)
        with col2:
            st.write("**Evaluation Type:**")
            st.write("📊 Simple Performance Test")
        with col3:
            if st.button("🚀 Start Evaluation", use_container_width=True):
                with st.spinner("🧠 Running RAG evaluation..."):
                    evaluation_result = simple_rag_evaluation(num_questions)
                    st.session_state['evaluation_results'] = evaluation_result
                    st.rerun()

        # Show existing results if available
        if 'evaluation_results' in st.session_state:
            result = st.session_state['evaluation_results']
            
            if "error" in result:
                st.error(f"❌ Evaluation failed: {result['error']}")
            else:
                st.success("✅ Evaluation completed!")
                
                # Metrics overview
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📈 Performance Metrics")
                    metrics = result.get("metrics", {})
                    
                    for metric, score in metrics.items():
                        st.markdown(f"""
                        <div class="metric-card">
                            <strong>{metric}:</strong> {score:.3f}
                            <br>
                            <small>{get_score_interpretation(score)}</small>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### 📊 Summary")
                    st.metric("Questions Tested", result.get("num_questions", 0))
                    st.metric("Documents Used", result.get("num_documents", 0))
                    
                    # Overall performance gauge
                    overall_score = metrics.get("Overall Performance", 0)
                    st.metric("Overall Score", f"{overall_score:.3f}", 
                             delta=f"{get_score_interpretation(overall_score)}")
                
                # Detailed results
                st.markdown("### 📋 Detailed Question Results")
                detailed_results = result.get("results", [])
                
                if detailed_results:
                    # Create DataFrame for display
                    df_data = []
                    for i, res in enumerate(detailed_results, 1):
                        df_data.append({
                            "Question #": i,
                            "Question": res["question"][:50] + "..." if len(res["question"]) > 50 else res["question"],
                            "Answer Length": len(res["answer"]),
                            "Length Score": f"{res['length_score']:.3f}",
                            "Relevance Score": f"{res['relevance_score']:.3f}",
                            "Overall Score": f"{res['overall_score']:.3f}"
                        })
                    
                    df = pd.DataFrame(df_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Show individual Q&A
                    with st.expander("📖 View Individual Questions & Answers"):
                        for i, res in enumerate(detailed_results, 1):
                            st.markdown(f"**Question {i}:** {res['question']}")
                            st.markdown(f"**Answer:** {res['answer'][:300]}{'...' if len(res['answer']) > 300 else ''}")
                            st.markdown(f"**Scores:** Length: {res['length_score']:.3f}, Relevance: {res['relevance_score']:.3f}, Overall: {res['overall_score']:.3f}")
                            st.markdown("---")
                    
                    # Download option
                    csv_data = pd.DataFrame(detailed_results).to_csv(index=False)
                    st.download_button(
                        label="📥 Download Detailed Results (CSV)",
                        data=csv_data,
                        file_name="rag_evaluation_results.csv",
                        mime="text/csv"
                    )

        # Reset evaluation
        if st.session_state.get('evaluation_results'):
            if st.button("🔄 Run New Evaluation"):
                if 'evaluation_results' in st.session_state:
                    del st.session_state['evaluation_results']
                st.rerun()

        # Evaluation explanation
        with st.expander("📚 How This Evaluation Works"):
            st.markdown("""
            **Simple RAG Evaluation Method:**
            
            1. **Document Retrieval**: Fetches documents from your RAG database
            2. **Test Questions**: Uses predefined FinTech-related questions
            3. **Answer Generation**: Gets answers from your RAG system
            4. **Scoring Metrics**:
               - **Length Score**: Based on answer completeness (longer = more detailed)
               - **Relevance Score**: Checks for FinTech keywords in answers
               - **Overall Score**: Average of length and relevance scores
            
            **Score Interpretation:**
            - 🟢 **Excellent (0.8+)**: High-quality, relevant answers
            - 🟡 **Good (0.6-0.8)**: Decent answers, some improvement needed
            - 🟠 **Needs Improvement (0.4-0.6)**: Answers lack detail or relevance
            - 🔴 **Poor (<0.4)**: Significant issues with answer quality
            
            *Note: This is a simplified evaluation. For more comprehensive analysis, consider using RAGas framework.*
            """)

    st.markdown('</div>', unsafe_allow_html=True)

# =============================================
# FOOTER
# =============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; opacity: 0.7; margin-top: 2rem; font-size: 1.1rem;">
    💸 <strong>FinTech RAG Chatbot</strong> - Powered by Google AI Studio & Milvus Vector Database
    <br>
    <small>Advanced RAG with Hypothetical Questions & Cross-Encoder Reranking</small>
</div>
""", unsafe_allow_html=True)