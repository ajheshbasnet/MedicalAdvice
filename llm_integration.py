import cv2
import base64
import numpy as np
from PIL import Image
from mistralai import Mistral
import google.generativeai as genai
import io
import re

class LLMProcessor:
    def __init__(self, mistral_api_key: str, genai_api_key: str):
        self.mistral_client = Mistral(api_key=mistral_api_key)
        genai.configure(api_key=genai_api_key)
        self.genai_model = genai.GenerativeModel('gemini-2.0-flash')

    def preprocess_image(self, image_bytes: bytes) -> str:
        # Convert bytes to a NumPy array
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        # Decode the image
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        # Denoising using morphological transformations
        kernel = np.ones((1, 1), np.uint8)
        denoised = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Deskewing the image
        coords = np.column_stack(np.where(denoised > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90
        else:
            angle = angle * -1

        (h, w) = denoised.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(denoised, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # Encode the processed image to base64
        _, buffer = cv2.imencode('.jpg', deskewed)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        return base64_image

    def extract_text_from_image(self, base64_image: str) -> str:
        ocr_response = self.mistral_client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{base64_image}"
            }
        )
        text_content = "\n\n".join(page.markdown for page in ocr_response.pages)
        return text_content

    def clean_text(self, text: str) -> str:
        text = text.replace("\n", "")
        text = re.sub(r"[^a-zA-Z0-9\s\(\)\.\-mg]", "", text)  # Remove unwanted characters
        text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
        return text

    def generate_response(self, cleaned_text: str) -> str:
        input_prompt = (
            f"{cleaned_text} I have a list of medicines along with their dosages. "
            "Based on these medicines, analyze what possible health issues I might be facing. "
            "Additionally, identify which aspects of my health I should focus on the most. "
            "Furthermore, suggest natural home remedies that can complement my prescribed medicines."
            "Provide details on their effectiveness, how to use them, and any precautions I should take. and dont give too much large output texts. give around 500 words. "
            "If the condition is serious, recommend consulting a doctor. "
            "Please ensure the response is detailed, structured, and informative. "
            "Conclude with a positive message like 'Get well soon' or 'Have a good day'."
        )
        response = self.genai_model.generate_content(input_prompt)
        return response.text
