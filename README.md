# GenAI Project Studio

A GenAI-powered application that helps developers find the right AI models and acts as an AI Architect to design software projects.

## Features

- **Perfect Model Finder**: Find the best LLM/SLM for your specific task based on benchmarks and constraints.
- **Project Architect**: Generate comprehensive technical blueprints, including tech stacks, file structures, and implementation guides.
- **History**: Save and retrieve generated project blueprints.

## Prerequisites

- Python 3.10+
- Google Cloud Project (for Firestore, Storage, BigQuery)
- Gemini API Key

## Setup

1. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

2. **Configuration**:
    - Create a `.env` file in the root directory:

      ```ini
      GEMINI_KEY=your_api_key_here
      BUCKET_NAME=your_bucket_name
      ```

    - Place your Google Cloud Service Account key as `service-account-key.json` in the root directory.

## Running the App

**Using Docker (Recommended):**

```bash
docker build -t genai-architect .
docker run -p 8080:8080 --env-file .env genai-architect
```

**Locally:**

1. Run `python server.py` to start the backend.
2. Run `streamlit run app.py` to start the frontend (if using Streamlit UI) or access `http://localhost:8080` for the HTML UI.
