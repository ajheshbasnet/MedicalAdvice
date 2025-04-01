import cv2
import base64
import numpy as np
from PIL import Image
from mistralai import Mistral
import google.generativeai as genai
import io, os
import re

class LLMProcessor:
    def __init__(self, mistral_api_key: str, genai_api_key: str):
        self.mistral_client = Mistral(api_key=mistral_api_key)
        genai.configure(api_key=genai_api_key)
        self.genai_model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25')

    def preprocess_image(self, image_bytes):
        """Process image bytes and return base64 encoded string"""
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Reduce noise while keeping edges
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

        # Adaptive Thresholding for binarization
        binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 15, 8)

        # Additional Denoising using Non-Local Means
        binary = cv2.fastNlMeansDenoising(binary, None, 30, 7, 21)

        # Invert colors (OCR works better with black text on white)
        inverted = cv2.bitwise_not(binary)

        # Deskewing to straighten text
        def deskew(img):
            coords = np.column_stack(np.where(img > 0))
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        deskewed = deskew(inverted)

        # Resize image for better text recognition
        resized = cv2.resize(deskewed, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # Convert processed image to base64
        _, buffer = cv2.imencode('.jpg', resized)
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
            f'''{cleaned_text}:
            Behave like Expert... I have a list of medicines along with their dosages. Extract only the medicine names and their corresponding dosages (if provided).

            Additional Instructions:
            Health Analysis: Based on the medicines, identify the possible health conditions I might be facing.

            Health Focus: Point out the most important areas of my health that need attention.

            Natural Remedies: Suggest simple home remedies that can help alongside my medications.

            Mention how to use them and any important precautions.

            If my condition seems serious, clearly say: "If your condition is serious, please visit a doctor immediately."

            Keep It Short and Practical: Don't explain too much. Just give it in short, clear sentences—not too little, just enough to understand.

            Closing Note: End with a positive message like "Get well soon!" or "Have a great day!"

            Keep it structured, informative, and easy to read. '''
        )
        response = self.genai_model.generate_content(input_prompt)
        return response.text