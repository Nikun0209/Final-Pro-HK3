import streamlit as st 
import sqlite3
import tempfile
import requests
import json
import uuid
import pdfplumber
from streamlit_option_menu import option_menu 
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from collections import defaultdict

# --- Initialize Qdrant ---
COLLECTION_NAME = "qdrant-sources"
client = QdrantClient(":memory:")

client.recreate_collection(
    COLLECTION_NAME,
    vectors_config={
        "text": models.VectorParams(size=384, distance=models.Distance.COSINE),
        "code": models.VectorParams(size=768, distance=models.Distance.COSINE),
    }
)

# --- Load models ---
text_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
code_model = SentenceTransformer("jinaai/jina-embeddings-v2-base-code", device="cpu")

# --- Connect to DB ---
def connect_to_db(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            
            temp_file.write(uploaded_file.getvalue())
            conn = sqlite3.connect(temp_file.name)
            cursor = conn.cursor()
            
            return conn, cursor
        
    except Exception as e:
        st.error(f"Error connecting to the database: {e}")
        return None, None

# --- Get table schema ---
def get_table_schema(cursor):
    try:
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        schema = {}
        
        for table in tables:
            
            cursor.execute(f"PRAGMA table_info({table});")
            columns = [col[1] for col in cursor.fetchall()]
            schema[table] = columns
            
        return schema
    
    except Exception as e:
        st.error(f"Error fetching schema: {e}")
        return {}

# --- Gemini SQL Generator ---
def generate_sql_query(question, api_key, schema):
    
    if not schema:
        st.warning("Schema is missing.")
        return None

    schema_text = "\n".join([f"{table}: {', '.join(cols)}" for table, cols in schema.items()])
    
    prompt = f"""Based on the following schema, convert the user question into an SQL query.

        Schema:
        {schema_text}

        Question:
        {question}

        If the question is not related to any table or column in the schema, reply: 'The question is not in the data.'
        Only return the SQL query if valid.
    """

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        
        if response.status_code == 200:
            
            result = response.json()
            reply = result["candidates"][0]["content"]["parts"][0]["text"].strip()
            
            if "the question is not in the data" in reply.lower():
                return "The question is not in the data."
            return reply.replace("```sql", "").replace("```", "").strip()
        else:
            st.error(f"Request failed: {response.status_code}")
            return None
        
    except Exception as e:
        st.error(f"Error calling Gemini API: {e}")
        return None

# --- Upload and Embed ---
def upload_code_to_qdrant(content, file_id=None, file_name=None):
    
    lines = content.strip().split('\n')
    code_snippets = [line for line in lines if line.strip()]
    text_embeddings = text_model.encode(code_snippets, batch_size=5)
    code_embeddings = code_model.encode(code_snippets, batch_size=5)

    points = []
    
    for i, (txt_emb, code_emb) in enumerate(zip(text_embeddings, code_embeddings)):
        points.append(models.PointStruct(
            id=str(uuid.uuid4()),
            vector={"text": txt_emb, "code": code_emb},
            payload={
                "original_text": code_snippets[i],
                "original_code": code_snippets[i],
                "file_id": file_id or str(uuid.uuid4()),
                "file_name": file_name or "unknown"
            }
        ))

    client.upsert(collection_name=COLLECTION_NAME, points=points)

# --- Search ---
def search_code_in_qdrant(query, top_k=5):
    
    query_vector = text_model.encode([query])[0]
    
    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=("text", query_vector),
        limit=top_k,
        with_payload=True
    )
    
    results = [{
        'score': hit.score,
        'text': hit.payload.get("original_text", ""),
        'code': hit.payload.get("original_code", ""),
        'file_name': hit.payload.get("file_name", "Unknown")
    } for hit in hits]
    
    return sorted(results, key=lambda x: -x['score'])


st.set_page_config(
    page_title="BenCodeX - Final Pro HK3",
    page_icon="🤖",
    layout="centered"
)

# --- Sidebar ---
with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=["SQL QUERY", "CODE SEARCH", "FILE SEARCH"],
        icons=["server", "file-code", "file-pdf"],
        menu_icon="menu-up",
        default_index=0
    )

# --- SQL QUERY ---
if selected == "SQL QUERY":
    
    st.header("🗟 CONVERT QUESTION TO SQL QUERY")
    
    GEMINI_API_KEY = "AIzaSyDOKOHKPqicPVamQ1VgI0NjUSkFfLHatTs"
    
    uploaded_file = st.file_uploader("Choose a database file (.db)", type=["db", "sqlite"])
    
    user_input = st.text_input("Enter your query question:")

    if uploaded_file:
        
        conn, cursor = connect_to_db(uploaded_file)
        
        if conn and cursor:
            
            schema = get_table_schema(cursor)
            
            if schema:
                
                st.write("📋 Tables in the database:", schema)
            
            if user_input:
                
                with st.spinner("🤖 Generating SQL query..."):
                    
                    sql_query = generate_sql_query(user_input, GEMINI_API_KEY, schema)
                    
                    if sql_query:
                        
                        st.markdown("✅ SQL query:")
                        st.code(sql_query, language='sql')
                        
                    else:
                        st.warning("Could not generate SQL query.")

# --- CODE SEARCH ---
elif selected == "CODE SEARCH":
    
    st.header("🔍 SEARCH SOURCE CODE WITH QDRANT")
    
    uploaded_code_file = st.file_uploader("📄 Upload source code file (.py)", type=["py"])
    
    if uploaded_code_file is not None:
        
        content = uploaded_code_file.read().decode("utf-8")
        
        st.text_area("📄 File content:", content, height=300)
        
        if content.strip():
            
            with st.spinner("🚀 Embedding and saving to Qdrant..."):
                
                upload_code_to_qdrant(content, file_name=uploaded_code_file.name)
                
                st.success("✅ Code saved to Qdrant!")
                
            st.markdown("---")
            
            st.subheader("🔍 Enter content to search in the code:")
            
            query = st.text_input("📝 Search query:")
            
            if query.strip():
                
                results = search_code_in_qdrant(query)
                
                if results:
                    
                    st.markdown(f"### 📄 Top {len(results)} results:")
                    
                    for i, res in enumerate(results, 1):
                        
                        st.markdown(f"**Result {i}:** (Score: {res['score']:.2f}) from `{res['file_name']}`")
                        
                        st.markdown(f"- **Text**: {res['text']}")
                        
                        st.code(res['code'], language='python')
                else:
                    st.info("❌ No results found.")
        else:
            st.warning("⚠️ File does not contain valid content!")

# --- FILE SEARCH ---
elif selected == "FILE SEARCH":
    
    st.header("📃 UPLOAD AND SEARCH PDF FILES")
    
    uploaded_pdfs = st.file_uploader("❇️ Upload PDF files", type=["pdf"], accept_multiple_files=True)

    if uploaded_pdfs:
        file_results, uploaded_files, success_files, skipped_files, no_text_files, failed_files = [], set(), [], [], [], []

        for pdf_file in uploaded_pdfs:
            
            try:                        
                file_id = str(uuid.uuid4())
                all_text = []

                # Giả sử `pdf_file` là đối tượng file dạng bytes
                with pdfplumber.open(pdf_file) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text().strip()
                        
                        if text:
                            all_text.append(text)

                if all_text:
                    
                    combined_text = "\n\n".join(all_text)
                    
                    upload_code_to_qdrant(combined_text, file_id=file_id, file_name=pdf_file.name)

                    if pdf_file.name not in uploaded_files:
                        
                        file_results.append({
                            'file_name': pdf_file.name,
                            'content': combined_text,
                            'score': None
                        })
                        
                        uploaded_files.add(pdf_file.name)
                        success_files.append(pdf_file.name)
                        
                    else:
                        skipped_files.append(pdf_file.name)
                else:
                    no_text_files.append(pdf_file.name)
                    
            except Exception as e:
                failed_files.append((pdf_file.name, str(e)))

        if success_files:
            st.success(f"✅ Successfully uploaded {len(success_files)} files.")

        if skipped_files:
            st.info(f"⚠️ Skipped (already uploaded): {', '.join(skipped_files)}")

        if no_text_files:
            st.warning(f"⚠️ No extractable text found in: {', '.join(no_text_files)}")

        if failed_files:
            for fname, error in failed_files:
                st.error(f"❌ Failed to process {fname}: {error}")

        st.markdown("---")
        
        st.subheader("🔍 Search content across uploaded PDFs:")
        
        query = st.text_input("🔎 Enter your question or keyword:")

        if query.strip():
            
            results = search_code_in_qdrant(query)

            score_threshold = 0.5
            filtered_results = [r for r in results if r['score'] >= score_threshold]

            grouped_results = defaultdict(list)
            
            for res in filtered_results:
                grouped_results[res['file_name']].append(res)

            if grouped_results:
                
                st.markdown(f"📄 Top {len(grouped_results)} relevant files:")
                
                uploaded_files_info = []

                for i, (file_name, group) in enumerate(grouped_results.items(), 1):
                    
                    best_result = max(group, key=lambda x: x['score'])
                    st.markdown(f"**Result {i}:** (Score: {best_result['score']:.2f}) from `{file_name}`")
                    st.markdown(f"- **Text**: {best_result['text']}")
                    st.code(best_result['code'], language='text')
                    uploaded_files_info.append((file_name, best_result['score']))

                st.markdown("✅ Found the following files:")
                
                for file_name, score in uploaded_files_info:
                    st.success(f"✅ {file_name} - Score: {score:.2f}")
            else:
                st.info("❌ No relevant results found (score below threshold).")


