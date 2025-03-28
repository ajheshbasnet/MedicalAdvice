from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from llm_integration import LLMProcessor

# Initialize FastAPI app
app = FastAPI()

# Setup Jinja2 for rendering HTML templates
templates = Jinja2Templates(directory="templates")

# Define API keys
MISTRAL_API_KEY = "PCkY1pM5ApzQLiwiE0eSOqPdkTZbb63j"
GENAI_API_KEY = "AIzaSyBCimoePb3Sbh3gnJKfFtj_XS6nvh-VkPE"

# Initialize LLM Processor
llm_processor = LLMProcessor(MISTRAL_API_KEY, GENAI_API_KEY)

@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    """ Serve the index.html template """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/inference")
async def ask_llm(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        base64_image = llm_processor.preprocess_image(image_bytes)
        extracted_text = llm_processor.extract_text_from_image(base64_image)
        cleaned_text = llm_processor.clean_text(extracted_text)
        response_text = llm_processor.generate_response(cleaned_text)
        return templates.TemplateResponse("index.html", {"request": {}, "response": response_text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
