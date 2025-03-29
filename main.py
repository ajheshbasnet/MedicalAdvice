from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from llm_integration import LLMProcessor

# Initialize FastAPI app
app = FastAPI()

# Setup Jinja2 for rendering HTML templates
templates = Jinja2Templates(directory="templates")

# Define API keys
MISTRAL_API_KEY = "MistralAPI"
GENAI_API_KEY = "GEMINIAPI"

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

        return JSONResponse(content={"response": response_text}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/additional_response")
async def additional_response():
    """
    Provides an additional structured response based on LLM analysis.
    """
    try:
        additional_info = """
        Health Suggestions:
        - Stay hydrated
        - Maintain a balanced diet
        - Get regular exercise
        - Follow your prescribed medications
        """
        return JSONResponse(content={"response": additional_info}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
