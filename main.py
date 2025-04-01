from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from llm_integration import LLMProcessor
import uuid

# Initialize FastAPI app
app = FastAPI()

# Setup Jinja2 for rendering HTML templates
templates = Jinja2Templates(directory="templates")

# Define API keys
MISTRAL_API_KEY = "5JkLJdGN5zKQP5XsCrwTFj1jT5hIApO8"
GENAI_API_KEY = "AIzaSyA-jC83f7PZXzP4XeGHO8gbFCo1aIPqeKI"

# Initialize LLM Processor
llm_processor = LLMProcessor(MISTRAL_API_KEY, GENAI_API_KEY)

@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    """ Serve the index.html template """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/inference")
async def ask_llm(file: UploadFile = File(...)):
    """
    Handles image upload, extracts text using OCR, and generates an AI response.
    """
    try:
        image_bytes = await file.read()
        base64_image = llm_processor.preprocess_image(image_bytes)
        extracted_text = llm_processor.extract_text_from_image(base64_image)
        cleaned_text = llm_processor.clean_text(extracted_text)
        response_text = llm_processor.generate_response(cleaned_text)

        # Fix formatting by replacing <br> with new lines
        response_text = response_text.replace("<br>", "\n").replace("**", "")

        return JSONResponse(
            content={
                "response": response_text,
                "extracted_text": extracted_text
            }, 
            status_code=200
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "OK"}