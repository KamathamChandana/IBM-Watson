# Install required libraries first:
# pip install transformers torch ibm-watson ibm-cloud-sdk-core pytesseract pillow

import pytesseract
from PIL import Image
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from ibm_watson import AssistantV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

# ------------------------------
# 1️⃣ OCR: Extract text from prescription image
# ------------------------------
def extract_text_from_image(image_path):
    text = pytesseract.image_to_string(Image.open(image_path))
    return text

# ------------------------------
# 2️⃣ Hugging Face NER: Extract drugs and dosages
# ------------------------------
def extract_medical_entities(text):
    model_name = "dmis-lab/biobert-base-cased-v1.1"  # Medical NER model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    entities = ner_pipeline(text)
    return entities

# ------------------------------
# 3️⃣ IBM Watson Assistant: Verify prescription
# ------------------------------
def verify_prescription_with_watson(text, api_key, url, assistant_id):
    authenticator = IAMAuthenticator(api_key)
    assistant = AssistantV2(version='2025-08-29', authenticator=authenticator)
    assistant.set_service_url(url)
    
    response = assistant.message_stateless(
        assistant_id=assistant_id,
        input={'text': f"Verify the following prescription: {text}"}
    ).get_result()
    
    return response

# ------------------------------
# 4️⃣ Main workflow
# ------------------------------
if __name__ == "__main__":
    # Path to prescription image (handwritten or typed)
    prescription_image = "prescription.jpg"
    
    # Extract text from image
    extracted_text = extract_text_from_image(prescription_image)
    print("Extracted Text:\n", extracted_text)
    
    # Extract medical entities (drugs, dosage)
    entities = extract_medical_entities(extracted_text)
    print("\nExtracted Medical Entities:")
    for ent in entities:
        print(ent)
    
    # Verify prescription with IBM Watson
    WATSON_API_KEY = "YOUR_WATSON_API_KEY"
    WATSON_URL = "YOUR_WATSON_URL"
    ASSISTANT_ID = "YOUR_ASSISTANT_ID"
    
    verification_result = verify_prescription_with_watson(
        extracted_text, WATSON_API_KEY, WATSON_URL, ASSISTANT_ID
    )
    
    print("\nIBM Watson Verification Result:")
    print(verification_result)
# Install required libraries first:
# pip install transformers torch ibm-watson ibm-cloud-sdk-core pytesseract pillow

import pytesseract
from PIL import Image
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from ibm_watson import AssistantV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

# ------------------------------
# 1️⃣ OCR: Extract text from prescription image
# ------------------------------
def extract_text_from_image(image_path):
    text = pytesseract.image_to_string(Image.open(image_path))
    return text

# ------------------------------
# 2️⃣ Hugging Face NER: Extract drugs and dosages
# ------------------------------
def extract_medical_entities(text):
    model_name = "dmis-lab/biobert-base-cased-v1.1"  # Medical NER model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    entities = ner_pipeline(text)
    return entities

# ------------------------------
# 3️⃣ IBM Watson Assistant: Verify prescription
# ------------------------------
def verify_prescription_with_watson(text, api_key, url, assistant_id):
    authenticator = IAMAuthenticator(api_key)
    assistant = AssistantV2(version='2025-08-29', authenticator=authenticator)
    assistant.set_service_url(url)
    
    response = assistant.message_stateless(
        assistant_id=assistant_id,
        input={'text': f"Verify the following prescription: {text}"}
    ).get_result()
    
    return response

# ------------------------------
# 4️⃣ Main workflow
# ------------------------------
if __name__ == "__main__":
    # Path to prescription image (handwritten or typed)
    prescription_image = "prescription.jpg"
    
    # Extract text from image
    extracted_text = extract_text_from_image(prescription_image)
    print("Extracted Text:\n", extracted_text)
    
    # Extract medical entities (drugs, dosage)
    entities = extract_medical_entities(extracted_text)
    print("\nExtracted Medical Entities:")
    for ent in entities:
        print(ent)
    
    # Verify prescription with IBM Watson
    WATSON_API_KEY = "YOUR_WATSON_API_KEY"
    WATSON_URL = "YOUR_WATSON_URL"
    ASSISTANT_ID = "YOUR_ASSISTANT_ID"
    
    verification_result = verify_prescription_with_watson(
        extracted_text, WATSON_API_KEY, WATSON_URL, ASSISTANT_ID
    )
    
    print("\nIBM Watson Verification Result:")
    print(verification_result)
