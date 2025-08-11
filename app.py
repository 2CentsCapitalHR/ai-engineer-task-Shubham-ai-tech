import gradio as gr
from corporate_agent import analyze_documents
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

def process_files(files):
    """Gradio wrapper function for the agent."""
    if files is None:
        return None, "Please upload .docx files.", {"error": "No files uploaded."}
        
    uploaded_file_paths = [file.name for file in files]
    output_path, summary = analyze_documents(uploaded_file_paths)
    
    status_message = "Analysis complete. Issues found are highlighted in the reviewed document."
    if output_path is None:
        # If no file was saved (either no issues or only low-severity ones)
        status_message = "No document with high-severity issues was found to review."
        
    # IMPORTANT: Return None for the file path if it doesn't exist
    return output_path, status_message, summary

# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # ... (your markdown)
    
    file_input = gr.File(
    label="Upload .docx Files",
    file_count="multiple",
    file_types=[".docx"]
    )# your file input definition
    submit_btn = gr.Button("Analyze Documents", variant="primary")
    
    gr.Markdown("---")
    gr.Markdown("## Results")
    
    with gr.Row():
        output_file = gr.File(label="Reviewed Document (with comments)")
        
        # Add a Textbox for status messages and move the JSON output here
        with gr.Column():
            status_text = gr.Textbox(label="Status", interactive=False)
            output_json = gr.JSON(label="Analysis Summary")
            
    submit_btn.click(
        fn=process_files,
        inputs=file_input,
        # Update the outputs to match the new layout
        outputs=[output_file, status_text, output_json],
        api_name="analyze"
    )

if __name__ == "__main__":
    demo.launch(share=True)