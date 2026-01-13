import os
import json
import time
import queue
import threading
import re
import warnings
import io
warnings.filterwarnings("ignore")
import ast
import pandas as pd
import numpy as np
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    print("⚠️ Qdrant Client not found or GRPC error. Vector DB features disabled.")
try:
    from googlesearch import search
    GOOGLE_SEARCH_AVAILABLE = True
except ImportError as e:
    GOOGLE_SEARCH_AVAILABLE = False
    print(f"⚠️ Google Search library not found: {e}. Web search features disabled.")
from flask import Flask, request, jsonify, Response, stream_with_context, render_template
from flask_cors import CORS
import google.generativeai as genai
from google.cloud import storage
from google.oauth2 import service_account
import google.auth
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

gemini_key = os.getenv("GEMINI_KEY")
if not gemini_key or "your_actual_gemini_api_key_here" in gemini_key:
    print("❌ Error: GEMINI_KEY is invalid or still set to placeholder in .env file. Get a key from https://aistudio.google.com/")
else:
    # Fix: Set standard environment variable for LangChain/Google libraries
    os.environ["GOOGLE_API_KEY"] = gemini_key
    os.environ["GEMINI_API_KEY"] = gemini_key

app = Flask(__name__)
CORS(app)

# --- Configuration & Auth ---
JSON_KEY_PATH = 'service-account-key.json' # Update this to your new key filename

# Define scopes to allow Service Account to access Gemini if API Key fails
SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/generative-language",
    "https://www.googleapis.com/auth/generative-language.retriever"
]

gcp_credentials = None
db = None
storage_client = None

if os.path.exists(JSON_KEY_PATH):
    try:
        gcp_credentials = service_account.Credentials.from_service_account_file(JSON_KEY_PATH, scopes=SCOPES)
        # Initialize Firebase
        if not firebase_admin._apps:
            cred = credentials.Certificate(JSON_KEY_PATH)
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        # Initialize Clients
        storage_client = storage.Client(credentials=gcp_credentials, project=gcp_credentials.project_id)
        print("✅ Google Cloud & Firebase Connected Successfully")
    except Exception as e:
        print(f"⚠️ Cloud Auth Skipped: {e}")
        print("ℹ️ App will run in 'Offline Mode' using Mock Data.")
else:
    # Fallback to Application Default Credentials (ADC) for Cloud Run
    print(f"⚠️ {JSON_KEY_PATH} not found. Attempting Application Default Credentials...")
    try:
        # Force scopes on default credentials (ADC)
        creds, project = google.auth.default(scopes=SCOPES)
        gcp_credentials = creds
        
        if not firebase_admin._apps:
            firebase_admin.initialize_app()
        db = firestore.client()
        storage_client = storage.Client(credentials=creds, project=project)
        print("✅ Google Cloud & Firebase Connected via Scoped ADC")
    except Exception as e:
        print(f"⚠️ Cloud Auth Failed: {e}")
        print("ℹ️ App will run in 'Offline Mode' using Mock Data.")

# Configure Gemini - Prefer API Key, fallback to Scoped Credentials
if os.getenv("GEMINI_KEY"):
    genai.configure(api_key=os.getenv("GEMINI_KEY"))
elif gcp_credentials:
    print("⚠️ GEMINI_KEY not found. Attempting to use Service Account for GenAI...")
    genai.configure(credentials=gcp_credentials)

# --- Custom Tools ---
def search_web(query):
    """Searches the web for the given query."""
    if not GOOGLE_SEARCH_AVAILABLE:
        return "Web search is disabled (missing dependencies)."
    try:
        print(f"🔎 Searching web for: {query}")
        # advanced=True returns objects with title, description, url
        results = search(query, num_results=3, advanced=True)
        output = []
        for result in results:
            output.append(f"Title: {result.title}\nLink: {result.url}\nSnippet: {result.description}")
            time.sleep(0.2) # Reduced delay for faster performance
        if not output:
            return "No search results found."
        return "\n\n".join(output)
    except Exception as e:
        return f"Search failed: {e}"

# --- Gen AI Models ---
generation_config = {
    "temperature": 0.2,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 2048,
    "response_mime_type": "application/json",
}

# Safety settings to prevent blocking of harmless technical content
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
    safety_settings=safety_settings
)

# --- Data Layer ---
_LEADERBOARD_CACHE = None
_CACHE_TIMESTAMP = 0
_CACHE_DURATION = 3600  # 1 hour

def load_leaderboard_data():
    """Load data from Google Cloud Storage with Caching"""
    global _LEADERBOARD_CACHE, _CACHE_TIMESTAMP
    
    if _LEADERBOARD_CACHE is not None and (time.time() - _CACHE_TIMESTAMP) < _CACHE_DURATION:
        return _LEADERBOARD_CACHE.copy()

    # 1. Try local file first (Robustness fix)
    df = pd.DataFrame()
    if os.path.exists("llm_leaderboard.csv"):
        print("📂 Loading leaderboard from local CSV file...")
        df = pd.read_csv("llm_leaderboard.csv")
        
        # Normalize column names to match expected format
        column_mapping = {
            'fullname': 'Model Name',
            'model_name': 'Model Name',
            'Model': 'Model Name',
            'Average': 'Average ⬆️'
        }
        df = df.rename(columns=column_mapping)
        df = df[(df['Official Providers']) & (df['Available on the hub']) & (~df['Flagged'])]
    elif storage_client:
        try:
            bucket_name = os.getenv("BUCKET_NAME", "genai-architect-data-2026")
            print(f"☁️ Attempting to load from Google Cloud Storage bucket: {bucket_name}...")
            bucket = storage_client.get_bucket(bucket_name)
            blob = bucket.blob("llm_leaderboard.csv")
            content = blob.download_as_string()
            df = pd.read_csv(io.BytesIO(content))
            # Filter for usable models
            column_mapping = {
                'fullname': 'Model Name',
                'model_name': 'Model Name',
                'Model': 'Model Name',
                'Average': 'Average ⬆️'
            }
            df = df.rename(columns=column_mapping)
            df = df[(df['Official Providers']) & (df['Available on the hub']) & (~df['Flagged'])]
            print("✅ Successfully loaded data from Google Cloud Storage.")
        except Exception as e:
            print(f"⚠️ Cloud Data Load Failed: {e}")
        
    # 3. Final Fallback: Mock Data (Ensures app runs even without CSV/Cloud)
    if df.empty:
        print("⚠️ Using Mock Leaderboard Data (Offline Mode)")
        df = pd.DataFrame([
            {'Model Name': 'Gemini 1.5 Pro', 'Official Providers': True, 'Available on the hub': True, 'Flagged': False, '#Params (B)': 0, 'Average ⬆️': 85.0, 'MMLU': 81.9, 'GSM8K': 92.0},
            {'Model Name': 'GPT-4o', 'Official Providers': True, 'Available on the hub': True, 'Flagged': False, '#Params (B)': 0, 'Average ⬆️': 88.0, 'MMLU': 88.7, 'GSM8K': 95.0},
            {'Model Name': 'Llama 3 70B', 'Official Providers': True, 'Available on the hub': True, 'Flagged': False, '#Params (B)': 70, 'Average ⬆️': 82.0, 'MMLU': 82.0, 'GSM8K': 85.0},
            {'Model Name': 'Llama 3 8B', 'Official Providers': True, 'Available on the hub': True, 'Flagged': False, '#Params (B)': 8, 'Average ⬆️': 65.0, 'MMLU': 65.0, 'GSM8K': 70.0},
            {'Model Name': 'Mistral 7B', 'Official Providers': True, 'Available on the hub': True, 'Flagged': False, '#Params (B)': 7, 'Average ⬆️': 60.0, 'MMLU': 60.0, 'GSM8K': 50.0},
            {'Model Name': 'Phi-3 Mini', 'Official Providers': True, 'Available on the hub': True, 'Flagged': False, '#Params (B)': 3.8, 'Average ⬆️': 68.0, 'MMLU': 68.0, 'GSM8K': 72.0},
        ])

    _LEADERBOARD_CACHE = df
    _CACHE_TIMESTAMP = time.time()
    return df

# --- Vector Search Logic (Simulated Vector DB) ---
class KnowledgeBase:
    def __init__(self):
        # Initialize Qdrant (In-memory for demo, use path="path/to/db" for persistence)
        if QDRANT_AVAILABLE:
            self.client = QdrantClient(":memory:")
            self.collection_name = "tech_blueprints"
            self._initialize_db()
        else:
            self.client = None

    def _get_embedding(self, text):
        # Using Google's embedding model as a proxy for Voyage-3.5
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']

    def _initialize_db(self):
        # Check if collection exists, if not create it
        if self.client and not self.client.collection_exists(self.collection_name):
            print("Initializing Qdrant Knowledge Base...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )
            
            # Seed with some "2026 Era" Tech Blueprints
            blueprints = [
                {"id": 1, "text": "Modern Web Stack: Next.js 15, React Server Components, Tailwind CSS, Supabase (PostgreSQL). Best for high-performance web apps."},
                {"id": 2, "text": "AI Agent Stack: Python 3.12, CrewAI, LangChain, Qdrant Vector DB, Gemini 1.5 Pro. Best for autonomous agent workflows."},
                {"id": 3, "text": "Mobile Stack: Flutter 4.0, Firebase, Riverpod. Best for cross-platform mobile apps with real-time data."},
                {"id": 4, "text": "Enterprise RAG: LlamaIndex, Voyage-3.5 Embeddings, Milvus, Docker. Best for large-scale document retrieval."},
            ]
            
            points = []
            for bp in blueprints:
                embedding = self._get_embedding(bp['text'])
                points.append(PointStruct(id=bp['id'], vector=embedding, payload={"info": bp['text']}))
            
            self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query):
        if not self.client:
            return []
        try:
            query_vec = genai.embed_content(
                model="models/text-embedding-004",
                content=query,
                task_type="retrieval_query"
            )['embedding']
            
            if hasattr(self.client, 'search'):
                hits = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vec,
                    limit=2
                )
                return [hit.payload['info'] for hit in hits]
            else:
                print("⚠️ QdrantClient missing 'search' method. Skipping RAG.")
                return []
        except Exception as e:
            print(f"⚠️ Vector DB Search Warning: {e}")
            # Return empty list so the app continues working without RAG
            return []

# --- Global KnowledgeBase Instance ---
_kb_instance = None

def get_knowledge_base():
    global _kb_instance
    if _kb_instance is None:
        try:
            _kb_instance = KnowledgeBase()
        except Exception as e:
            print(f"⚠️ KnowledgeBase Initialization Failed: {e}")
            return None
    return _kb_instance

# --- LangGraph-style Workflow Nodes ---

class ProjectArchitect:
    def __init__(self):
        self.kb = get_knowledge_base()

    def run_crew(self, user_task, max_params, status_queue=None):
        """Orchestrates the Agents using a Direct Sequential Chain (Robust Alternative)."""
        try:
            # 1. Retrieve relevant tech stacks from Qdrant
            context_docs = []
            if self.kb:
                context_docs = self.kb.search(user_task)
            
            if not context_docs:
                context_docs = ["No specific blueprints found in knowledge base. Use general knowledge and web search."]
                
            context_str = "\n".join(context_docs)

            # --- Step 1: Requirements Analyst ---
            print("🚀 Starting Analyst...")
            if status_queue: status_queue.put({"type": "log", "content": "🕵️ Requirements Analyst is analyzing the request..."})
            
            analyst_prompt = f"""
            You are a Principal Software Architect and Product Manager.
            Analyze this project request: "{user_task}"
            
            Your goal is to turn vague or specific inputs into a concrete technical specification.
            
            1. **Refine the Idea**: If the request is vague (e.g., "make a chat app"), define a modern, specific USP (Unique Selling Point).
            2. **Project Classification**: Is it a Web App, Mobile App, Data Pipeline, Automation Script, or Enterprise System?
            3. **Key Technical Challenges**: Identify the top 3 difficult parts (e.g., "Real-time latency", "Data privacy", "Context window limits").
            4. **Implicit Requirements**: What did the user forget? (e.g., Auth, Database, Hosting).
            
            Output a structured summary that a Tech Lead would appreciate.
            """
            _resp = model.generate_content(analyst_prompt)
            analyst_response = _resp.text if _resp.parts else "Analysis unavailable."
            if status_queue: status_queue.put({"type": "log", "content": f"Analyst Summary: {analyst_response[:100]}..."})

            # --- Step 2: Tech Stack Architect ---
            print("🚀 Starting Architect...")
            if status_queue: status_queue.put({"type": "log", "content": "🏗️ Tech Stack Architect is searching for tools..."})
            
            # Manual Tool Use (More robust than Agent tool use)
            search_query = f"best open source tech stack for {user_task} 2025"
            search_results = search_web(search_query)
            
            architect_prompt = f"""
            You are a CTO / Senior Tech Lead.
            
            **Input Context:**
            User Request: "{user_task}"
            Analyst Analysis:
            {analyst_response}
            
            Search Results (for tools & links):
            {search_results}
            
            **Goal:** Design the optimal technology stack.
            
            **Instructions:**
            1. **Select the Stack**: Choose the best tools for the job. Prioritize Modern, Stable, and Open Source tools.
            2. **Justification**: Briefly explain *why* you chose these tools over competitors (e.g., "FastAPI over Flask because of async support for AI").
            3. **Architecture**: Define the data flow (Frontend -> API -> LLM -> DB).
            4. **Bill of Materials**: List every tool, library, and service needed.
            
            Constraint: If recommending an LLM, choose one with < {max_params}B parameters.
            """
            _resp = model.generate_content(architect_prompt)
            architect_response = _resp.text if _resp.parts else "Architecture recommendation unavailable."
            if status_queue: status_queue.put({"type": "log", "content": "Architect has designed the stack."})

            # --- Step 3: Process Engineer ---
            print("🚀 Starting Engineer...")
            if status_queue: status_queue.put({"type": "log", "content": "✍️ Technical Writer is compiling the blueprint..."})
            
            engineer_prompt = f"""
            You are a Senior Staff Engineer and Technical Writer.
            
            **Goal:** Write a comprehensive, execution-ready Project Blueprint.
            
            **Input:**
            {architect_response}
            
            **Output Format (Markdown):**
            
            # 🚀 Project Blueprint: [Project Name]
            
            ## 🎯 Executive Summary
            (A pitch-perfect summary of what we are building and why it rocks.)
            
            ## 🏗️ System Architecture
            (Describe the high-level design and data flow.)
            
            ## 🛠️ Tech Stack & Bill of Materials
            (Table: Tool | Category | Purpose | Link)
            
            ## 📂 Project Structure
            (A full file tree structure. Example:)
            ```text
            project-name/
            ├── backend/
            │   ├── main.py
            │   └── ...
            └── ...
            ```
            
            ## 👣 Implementation Guide
            **Phase 1: Environment Setup**
            (Commands to install dependencies. Provide the content for `requirements.txt` or `package.json`.)
            
            **Phase 2: Core Logic Implementation**
            (Don't just say "write code". **Provide the actual Python/JS code** for the most critical function of the app. e.g., the AI inference loop or the WebSocket handler.)
            
            **Phase 3: UI/Frontend**
            (Key components and how they connect.)
            
            ## 🚀 Deployment & Scale
            (Docker, Cloud Run, or Vercel instructions.)
            
            Make it look like a premium technical document. Use formatting, emojis, and clear code blocks.
            **IMPORTANT: Output ONLY raw Markdown. Do NOT wrap in JSON or code blocks.**
            """
            
            # Override config to allow raw text output (Markdown) instead of JSON
            text_generation_config = {
                "temperature": 0.2,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192, # Increased for full blueprints
                "response_mime_type": "text/plain"
            }
            
            _resp = model.generate_content(engineer_prompt, generation_config=text_generation_config)
            result = _resp.text if _resp.parts else "Roadmap generation failed."
            
            # Clean up wrapping markdown code blocks while preserving internal code blocks
            result = result.strip()
            if result.startswith("```markdown"):
                result = result[11:]
                if result.endswith("```"):
                    result = result[:-3]
            elif result.startswith("```"):
                result = result[3:]
                if result.endswith("```"):
                    result = result[:-3]
            result = result.strip()
            
            # Robustness: Handle case where LLM returns JSON despite instructions
            # Use regex to find JSON object if it's embedded in text
            match = re.search(r'\{.*\}', result, re.DOTALL)
            if match:
                try:
                    potential_json = match.group(0)
                    parsed = json.loads(potential_json)
                    # Extract content from common keys
                    for key in ["blueprint", "markdown_report", "report", "content"]:
                        if key in parsed:
                            result = parsed[key]
                            break
                except json.JSONDecodeError:
                    pass
            
            print("✅ Process Finished")

            final_output = {
                "markdown_report": str(result)
            }
            
            if status_queue:
                status_queue.put({"type": "result", "data": final_output})
            return final_output
        except Exception as e:
            print(f"❌ Error in run_crew: {e}")
            if status_queue:
                status_queue.put({"type": "error", "content": str(e)})
            return {"markdown_report": f"Error: {str(e)}"}

# --- API Routes ---

@app.route('/api/find_model', methods=['POST'])
def find_model():
    try:
        data = request.json
        task = data.get('task')
        max_params = float(data.get('max_params', 15))
        top_k = int(data.get('top_k', 6))
        
        if not task:
            return jsonify({"error": "Task is required"}), 400

        df = load_leaderboard_data()
        if df.empty:
            return jsonify({"error": "Leaderboard data unavailable. Check Google Cloud Storage."}), 500

        # 0. Classify Task Type (LLM/SLM vs ML/DL)
        classification_prompt = f"""
        Analyze this machine learning task: "{task}"
        Determine if this is best solved by:
        1. "LLM": Large/Small Language Models (Text generation, summarization, coding, chat, reasoning).
        2. "ML_DL": Specific Machine Learning or Deep Learning models (Image classification, Object detection, Tabular regression, Clustering, Audio processing, Time series).
        
        Return JSON: {{ "category": "LLM" }} or {{ "category": "ML_DL" }}
        """
        category = "LLM" # Default
        try:
            resp = model.generate_content(classification_prompt)
            text = resp.text.replace("```json", "").replace("```", "").strip()
            cat_data = json.loads(text)
            category = cat_data.get("category", "LLM")
        except:
            pass

        if category == "ML_DL":
            # Handle ML/DL/Vision/Audio tasks via Generative Recommendation
            rec_prompt = f"""
            Task: "{task}"
            Target Audience: Freshers, Developers, and Engineers.
            Suggest {top_k} best state-of-the-art Machine Learning or Deep Learning models/algorithms for this specific task.
            Prioritize models that are:
            1. Open Source and Accessible (Hugging Face, GitHub).
            2. Well-documented and suitable for implementation.
            3. Industry standard (e.g., YOLO for vision, Whisper for audio).
            
            Return a JSON object with a key "models" containing a list of objects.
            Each object must have these exact keys:
            - "Model Name": Name of the model (e.g., "YOLOv8", "ResNet-50", "XGBoost", "Whisper").
            - "#Params (B)": Approximate parameter count in Billions (number) or 0 if not applicable.
            - "Average ⬆️": A hypothetical performance score (0-100) based on current SOTA standards.
            - "Architecture": e.g., "CNN", "Transformer", "Gradient Boosting".
            - "characteristics": Key features (e.g., "Real-time", "High Accuracy").
            - "purpose": Why it fits this task.
            - "url": Link to paper or repo.
            - "usage": Practical Python code snippet (using sklearn, pytorch, tensorflow, or huggingface) to load/inference.
            
            Output ONLY valid JSON.
            """
            try:
                resp = model.generate_content(rec_prompt)
                text_resp = resp.text.replace("```json", "").replace("```", "").strip()
                match = re.search(r'\{.*\}', text_resp, re.DOTALL)
                if match: text_resp = match.group(0)
                data = json.loads(text_resp)
                return jsonify({
                    "benchmarks": ["Accuracy", "SOTA Score"],
                    "models": data.get("models", [])
                })
            except Exception as e:
                print(f"ML/DL Gen failed: {e}. Falling back to LLM logic.")

        # 1. Get Benchmarks via Gemini
        prompt = f"""
        Analyze this task: "{task}"
        Which of these benchmarks are most relevant? ['MMLU', 'TRUTHFULQA', 'Hellaswag', 'GSM8K', 'ARC', 'Winogrande', 'MBPP', 'HumanEval']
        Return a Python list of strings. Example: ['MMLU', 'GSM8K']
        """
        try:
            resp = model.generate_content(prompt)
            if not resp.parts:
                raise ValueError("Blocked or empty response from AI")
            text = resp.text.replace("```python", "").replace("```", "").strip()
            match = re.search(r'\[.*?\]', text, re.DOTALL)
            benchmarks = ast.literal_eval(match.group(0)) if match else []
            if not isinstance(benchmarks, list): raise ValueError
        except:
            benchmarks = ["Average ⬆️"]

        # 2. Filter & Sort
        valid_benchmarks = [b for b in benchmarks if b in df.columns]
        if not valid_benchmarks: valid_benchmarks = ["Average ⬆️"]

        # Ensure numeric columns for calculation
        cols_to_numeric = valid_benchmarks + ['#Params (B)', 'Average ⬆️']
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['Score'] = df[valid_benchmarks].mean(axis=1)
        
        # Filter by size (SLM support)
        df_filtered = df[df['#Params (B)'] <= max_params]
        if df_filtered.empty:
            # Fallback if no models fit size constraint (show smallest available)
            df_filtered = df.sort_values('#Params (B)').head(5)
        
        # Get Top K Models
        top_models_df = df_filtered.sort_values('Score', ascending=False).head(top_k)
        
        models_list = []
        for _, row in top_models_df.iterrows():
            m_dict = row.replace({np.nan: None}).to_dict()
            models_list.append(m_dict)
            
        # Enrich with Gemini (Characteristics, Purpose, URL, Usage)
        model_names = [m['Model Name'] for m in models_list]
        enrich_prompt = f"""
        Task: "{task}"
        Models: {model_names}
        Target Audience: Developers and Engineers.

        For EACH model in the list, provide technical details.
        Return a JSON object with a key "models" containing a list of objects.
        Each object must have:
        - "model_name": The exact name from the input list.
        - "characteristics": Key technical features (e.g., Context window, GQA, MoE).
        - "purpose": Why it fits this specific task.
        - "url": Official link or Hugging Face URL.
        - "deployment": Where to use/deploy (e.g., "Ollama", "Hugging Face Inference", "vLLM", "LM Studio").
        - usage: A concise, working Python code snippet (using transformers/langchain) to run inference.

        Output ONLY valid JSON.
        """
        try:
            resp = model.generate_content(enrich_prompt)
            text_resp = resp.text.replace("```json", "").replace("```", "").strip()
            # Robust extraction: Find the JSON object/list if extra text is present
            match = re.search(r'(\{|\[).*(\}|\])', text_resp, re.DOTALL)
            if match:
                text_resp = match.group(0)
            enrichment_data = json.loads(text_resp)
            enrichment_list = enrichment_data.get("models", []) if isinstance(enrichment_data, dict) else []
            
            # Create lookup map
            enrichment_map = {item.get("model_name"): item for item in enrichment_list}
        except Exception as e:
            print(f"Enrichment failed: {e}")
            enrichment_map = {}

        # Merge enrichment data
        for m in models_list:
            name = m['Model Name']
            info = enrichment_map.get(name)
            # Fuzzy match fallback
            if not info:
                 for k, v in enrichment_map.items():
                     if k and (k in name or name in k):
                         info = v
                         break
            
            if info:
                m.update(info)
            
        return jsonify({
            "benchmarks": valid_benchmarks,
            "models": models_list
        })
    except Exception as e:
        print(f"Error in find_model: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/generate', methods=['POST'])
def generate_project():
    try:
        data = request.json
        task = data.get('task')
        max_params = float(data.get('max_params', 15))
        
        if not task:
            return jsonify({"error": "Task is required"}), 400
            
        def generate():
            q = queue.Queue()
            architect = ProjectArchitect()
            
            # Run CrewAI in a separate thread
            thread = threading.Thread(target=architect.run_crew, args=(task, max_params, q))
            thread.start()
            
            while True:
                try:
                    # Wait for data from the queue
                    item = q.get(timeout=1)
                    yield f"data: {json.dumps(item)}\n\n"
                    
                    if item["type"] == "result":
                        break
                except queue.Empty:
                    # Send keep-alive to prevent browser timeout
                    yield ": keep-alive\n\n"
                    if not thread.is_alive():
                        break
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
                    break

        return Response(stream_with_context(generate()), mimetype='text/event-stream')

    except Exception as e:
        print(f"Error in generate_project: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/save_project', methods=['POST'])
def save_project():
    try:
        data = request.json
        project_data = data.get('project_data')
        
        if not db: return jsonify({"error": "Database offline"}), 503
        if not project_data: return jsonify({"error": "No project data provided"}), 400
        
        doc_ref = db.collection('generated_projects').document()
        save_data = project_data.copy()
        save_data['timestamp'] = firestore.SERVER_TIMESTAMP
        doc_ref.set(save_data)
        
        return jsonify({"message": "Project saved successfully", "id": doc_ref.id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    try:
        if not db:
            return jsonify({"error": "Database not connected (Offline Mode)"}), 503
        
        # Fetch last 10 projects
        docs = db.collection('generated_projects').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(10).stream()
        history = []
        for doc in docs:
            data = doc.to_dict()
            # Handle timestamp serialization
            if 'timestamp' in data and data['timestamp']:
                if hasattr(data['timestamp'], 'isoformat'):
                    data['timestamp'] = data['timestamp'].isoformat()
                else:
                    data['timestamp'] = str(data['timestamp'])
            data['id'] = doc.id
            history.append(data)
        
        return jsonify(history)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/history', methods=['DELETE'])
def clear_history():
    try:
        if not db:
            return jsonify({"error": "Database not connected (Offline Mode)"}), 503
        
        # Batch delete (Firestore)
        docs = db.collection('generated_projects').stream()
        deleted_count = 0
        for doc in docs:
            doc.reference.delete()
            deleted_count += 1
            
        return jsonify({"message": f"Cleared {deleted_count} projects."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/project/<project_id>', methods=['DELETE'])
def delete_project(project_id):
    try:
        if not db:
            return jsonify({"error": "Database not connected (Offline Mode)"}), 503
        
        db.collection('generated_projects').document(project_id).delete()
        return jsonify({"message": "Project deleted successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Prioritize BACKEND_PORT (from start.sh) over PORT (Cloud Run default)
    port = int(os.environ.get('BACKEND_PORT', os.environ.get('PORT', 8080)))
    print(f"🚀 Backend Server running on http://0.0.0.0:{port}")
    app.run(debug=True, host='0.0.0.0', port=port, exclude_patterns=['*/venv/*', '*/__pycache__/*'])