AI Corporate Agent for ADGM Document Compliance
This project is an AI-powered legal assistant designed to automatically review legal documents for compliance with Abu Dhabi Global Market (ADGM) regulations.

It helps users ensure their paperwork is correct before submission by checking for missing documents, identifying red flags, and providing a detailed analysis report.

How It Works
The process is simple and automated:

Upload Documents: The user uploads one or more legal documents (in .docx format) through a simple web interface.

Identify and Check: The system first identifies the type of each document (e.g., "Articles of Association"). It then compares the uploaded files against a predefined checklist to see if any required documents are missing for the legal process (e.g., Company Incorporation).

AI Analysis (RAG): Each document is broken down into chunks. The system uses a Retrieval-Augmented Generation (RAG) pipeline. For each chunk, it retrieves relevant ADGM legal rules from a specialized vector database and sends both the document chunk and the rules to the Google Gemini LLM.

Find Red Flags: The AI analyzes the text to find compliance issues, such as incorrect legal jurisdiction, ambiguous language, or missing clauses, based on the provided ADGM rules.

Generate Results: The agent produces two key outputs:

A reviewed .docx file with AI comments inserted directly into the text where issues were found.

A JSON summary report that lists all missing documents and details every issue identified across all files.

Key Features
Document Checklist Verification: Automatically checks if all mandatory documents for a specific process (like company incorporation) have been uploaded.

AI-Powered Red Flag Detection: Uses a RAG pipeline with Google Gemini to find compliance issues with high accuracy, grounded in actual ADGMs regulations.

Automated Document Annotation: Generates a new .docx file with AI feedback inserted as bolded text, making it easy to see what needs to be fixed.

Structured Reporting: Outputs a clean JSON file that summarizes the entire analysis, perfect for record-keeping or further processing.

Simple Web Interface: Built with Gradio for an easy-to-use, interactive demonstration.

Technology Stack
Backend: Python

AI Framework: LangChain

Language Model (LLM): Google Gemini (gemini-1.5-flash)

Embeddings & Vector Store: HuggingFace Transformers & FAISS

Web UI: Gradio

File Handling: python-docx

Setup and Installation
To run this project locally, follow these steps:

Clone the Repository:

git clone <your-repository-url>
cd <repository-folder>

Install Dependencies:
Install all required Python packages using the requirements.txt file.

pip install -r requirements.txt

Set Up Environment Variables:

Create a file named .env in the root directory.

Add your Google API key to this file:

GOOGLE_API_KEY="your_google_api_key_here"

Run the Application:
Execute the app.py script to launch the Gradio web interface.

python3 app.py

The terminal will provide a local URL to access the application.

How to Use the Agent
Open the application using the URL provided when you run app.py.

Drag and drop one or more .docx files into the "Upload Files" box.

Click the "Analyze Documents" button.

Wait for the analysis to complete. The results will appear on the right:

A downloadable .docx file with AI comments.

A JSON object summarizing the findings.
