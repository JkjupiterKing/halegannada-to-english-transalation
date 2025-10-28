import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import os
import google.generativeai as genai
import openai # Added for DeepSeek
import traceback
import time
import json
import base64
from PIL import Image
from io import BytesIO
import random
import sys
sys.stdout.reconfigure(line_buffering=True)

app = Flask(__name__)
CORS(app)

# Configure Gemini API
# --- Security Best Practice: Load API key from environment variable ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# It's critical to check if the key was actually loaded.
if not GEMINI_API_KEY:
    # This error will be printed to the console when the server starts.
    print("Error: GEMINI_API_KEY environment variable not set.")
    # In a production environment, you might want to exit or handle this more gracefully.
    # For this application, we'll allow it to start but Gemini features will fail.

genai.configure(api_key=GEMINI_API_KEY)

# Configure DeepSeek API via OpenRouter
DEEPSEEK_API_KEY = "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" # This is an OpenRouter key
deepseek_client = openai.OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://openrouter.ai/api/v1")

# Rate limiting and retry configuration
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 10  # seconds
MAX_RETRY_DELAY = 120    # seconds

def exponential_backoff(attempt):
    """Calculate delay with exponential backoff and jitter"""
    delay = min(MAX_RETRY_DELAY, INITIAL_RETRY_DELAY * (2 ** attempt))
    jitter = random.uniform(0, 0.1 * delay)  # 10% jitter
    return delay + jitter

def retry_with_backoff(func, *args, **kwargs):
    """Retry function with exponential backoff"""
    last_exception = None
    
    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            
            # Check if it's a quota error (common for HTTP 429)
            # For DeepSeek/OpenAI, APIError often includes status codes
            status_code = getattr(e, 'status_code', None)
            if "429" in str(e) or "quota" in str(e).lower() or status_code == 429:
                if attempt < MAX_RETRIES - 1:
                    delay = exponential_backoff(attempt)
                    print(f"Rate limit hit or temporary API error. Retrying in {delay:.2f} seconds... (Attempt {attempt + 1}/{MAX_RETRIES})")
                    time.sleep(delay)
                else:
                    print(f"Max retries reached for rate limit error: {str(e)}")
                continue # Continue to next attempt or exit loop if max retries reached
            else:
                # If it's not a known retryable error, raise immediately
                print(f"Non-retryable API error: {str(e)}")
                raise e
    
    raise Exception(f"Max retries ({MAX_RETRIES}) exceeded. Last error: {str(last_exception)}")

# Initialize the Gemini model - using the latest pro model that supports vision
try:
    gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest') # Renamed to gemini_model for clarity
    print("Successfully initialized Gemini model")
except Exception as e:
    print(f"Error initializing Gemini model: {str(e)}")
    traceback.print_exc()
    gemini_model = None # Ensure it's None if init fails

def process_image(image_data):
    """Process image data: extract Halegannada text using Gemini, then translate using DeepSeek."""
    try:
        print("\n=== Starting Image Processing ===")
        print("Step 1: Checking image data...")

        if not image_data:
            print("Error: No image data received")
            return {"error": "Error: No image data received"}

        try:
            if isinstance(image_data, str):
                if image_data.startswith('data:image'):
                    print("Converting base64 to bytes...")
                    try:
                        base64_data = image_data.split(',')[1]
                        image_bytes = base64.b64decode(base64_data)
                        print("Base64 conversion successful")
                    except Exception as e:
                        print(f"Base64 decode error: {str(e)}")
                        return {"error": f"Error decoding image: {str(e)}"}
                else:
                    print("Error: Invalid image data format")
                    return {"error": "Error: Invalid image data format"}
            else:
                image_bytes = image_data  # Assuming it's already bytes

            print("Step 2: Opening image...")
            try:
                image = Image.open(BytesIO(image_bytes))
                print(f"Image opened successfully. Format: {image.format}, Size: {image.size}, Mode: {image.mode}")
            except Exception as e:
                print(f"Error opening image: {str(e)}")
                return {"error": f"Error opening image: {str(e)}"}

            print("Step 3: Converting image format...")
            try:
                if image.mode in ('RGBA', 'P'):  # Convert to RGB if necessary
                    image = image.convert('RGB')
                    print("Converted to RGB mode")
            except Exception as e:
                print(f"Error converting image format: {str(e)}")
                return {"error": f"Error converting image format: {str(e)}"}

            print("Step 4: Resizing image if needed...")
            try:
                max_size = 768  # Example max size
                if max(image.size) > max_size:
                    ratio = max_size / max(image.size)
                    new_size = tuple(int(dim * ratio) for dim in image.size)
                    image = image.resize(new_size, Image.Resampling.LANCZOS)  # Using LANCZOS for quality
                    print(f"Image resized to {new_size}")
            except Exception as e:
                print(f"Error resizing image: {str(e)}")
                return {"error": f"Error resizing image: {str(e)}"}

            print("Step 5: Preparing for Gemini API to extract text...")
            gemini_ocr_prompt = """
            Task: Extract Halegannada (Old Kannada) text from this image.
            Instructions:
            1. Look for Halegannada text in the image.
            2. Return ONLY the extracted Halegannada text.
            3. Preserve any line breaks or formatting in the extracted text.
            If no Halegannada text is found, respond with "No Halegannada text detected in image"
            """

            if not gemini_model:
                return {"error": "Error: Gemini model not initialized."}

            print("Step 6: Calling Gemini API for OCR...")
            try:
                img_byte_arr = BytesIO()
                # Determine image format; default to JPEG if not available or unsupported by PIL for saving
                save_format = image.format if image.format and image.format.upper() in ['JPEG', 'PNG', 'WEBP'] else 'JPEG'
                mime_type = f'image/{save_format.lower()}'

                image.save(img_byte_arr, format=save_format)
                img_byte_arr = img_byte_arr.getvalue()

                gemini_response = retry_with_backoff(
                    lambda: gemini_model.generate_content([
                        gemini_ocr_prompt,
                        {"mime_type": mime_type, "data": img_byte_arr}
                    ])
                )
                print("Received response from Gemini API for OCR")

                if not gemini_response or not hasattr(gemini_response, 'text'):
                    print("Error: Invalid or empty response from Gemini OCR")
                    return {"error": "Error: Could not extract text using Gemini"}

                extracted_text = gemini_response.text.strip()
                print(f"Text extracted by Gemini: {extracted_text[:200]}...")

                if not extracted_text or "No Halegannada text detected" in extracted_text:
                    return {"extracted_text": "No Halegannada text detected in image", "english": ""}

                print("Step 7: Translating extracted text using DeepSeek API...")
                english_translation = get_english_translation(extracted_text)  # This now uses DeepSeek

                if "Translation error" in english_translation or "not available" in english_translation:
                    print(f"DeepSeek translation failed for extracted text: {english_translation}")
                    return {"extracted_text": extracted_text, "error": "Error translating extracted text with DeepSeek"}

                print(f"DeepSeek Translation successful. Result: {english_translation[:100]}...")
                return {"extracted_text": extracted_text, "english": english_translation}

            except Exception as e:
                print(f"Gemini API or subsequent DeepSeek call error: {str(e)}")
                traceback.print_exc()
                return {"error": f"Error during image processing pipeline: {str(e)}"}

        except Exception as e:  # Catches errors in image pre-processing
            print(f"Image pre-processing error: {str(e)}")
            traceback.print_exc()
            return {"error": f"Image pre-processing error: {str(e)}"}

    except Exception as e:  # Broadest catch for any unexpected error in process_image
        print("\n=== Error in process_image ===")
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return {"error": f"Overall error in process_image: {str(e)}"}

def get_kannada_translation(word):
    """Get Kannada translation from Dictionary.pkl"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dictionary_path = os.path.join(current_dir, 'assets', 'Dictionary.pkl')
        
        if not os.path.exists(dictionary_path):
            print(f"Error: Dictionary.pkl not found at {dictionary_path}")
            return word # Return original word if dictionary is missing
            
        with open(dictionary_path, 'rb') as f:
            meanings = pickle.load(f)
            return meanings.get(word, word)
    except Exception as e:
        print(f"Error loading dictionary: {e}")
        traceback.print_exc()
        return word

def get_english_translation(text_to_translate):
    """Get English translation using DeepSeek via OpenRouter API"""
    if not text_to_translate:
        return "No text provided for translation."
    print(f"Attempting OpenRouter (DeepSeek) translation for: {text_to_translate[:100]}...")
    
    def _translate_with_openrouter_deepseek():
        try:
            response = deepseek_client.chat.completions.create(
                model="deepseek/deepseek-chat", 
                messages=[
                    {"role": "system", "content": "You are an expert translator. Translate the given text to English concisely and precisely. Provide only the direct translation, without any extra explanations, commentary, or conversational filler."},
                    {"role": "user", "content": f"Translate the following Halegannada (Old Kannada) or Modern Kannada text to English. Be brief and precise:\n\n{text_to_translate}"}
                ],
                stream=False,
                max_tokens=1000,  # Slightly reduced, but prompt is key
                temperature=0.5   # Lowered for more precise, less verbose output
            )
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                translated_text = response.choices[0].message.content.strip()
                print("OpenRouter (DeepSeek) translation successful.")
                return translated_text
            else:
                print("OpenRouter (DeepSeek) API returned an empty or invalid response.")
                return "Translation not available from OpenRouter (DeepSeek) (empty response)"
        except openai.APIError as e:
            print(f"OpenRouter (DeepSeek) APIError: {e.status_code} - {e.message}")
            # Check for model not found error, which might indicate the model string is wrong for OpenRouter
            if e.status_code == 404 and "model_not_found" in str(e.code).lower():
                 print("Model 'deepseek/deepseek-chat' might be incorrect for OpenRouter. Please verify the model identifier on OpenRouter.")
                 # Return a more specific error message
                 return "Translation failed: Model not found on OpenRouter. Check model identifier."
            raise # Re-raise to be caught by retry_with_backoff
        except Exception as e:
            print(f"Generic error during OpenRouter (DeepSeek) translation: {str(e)}")
            raise # Re-raise to be caught by retry_with_backoff

    try:
        return retry_with_backoff(_translate_with_openrouter_deepseek)
    except Exception as e:
        error_message = f"OpenRouter (DeepSeek) translation error after retries: {str(e)}"
        print(error_message)
        if "Max retries" in str(e) and ("429" in str(e) or "quota" in str(e).lower()):
             return "Translation failed due to OpenRouter API rate limits after multiple retries."
        if "Model not found on OpenRouter" in str(e): # Propagate specific model not found error
            return "Translation failed: Model not found on OpenRouter. Check model identifier."
        return f"Translation error: OpenRouter (DeepSeek) API unavailable or failed after retries."

def get_hosa_kannada_translation_api(text_to_translate):
    """Get Hosa Kannada translation using DeepSeek via OpenRouter API."""
    if not text_to_translate:
        return "No text provided for translation."
    print(f"Attempting OpenRouter (DeepSeek) Hosa Kannada translation for: {text_to_translate[:100]}...")

    def _translate_with_openrouter_deepseek():
        try:
            response = deepseek_client.chat.completions.create(
                model="deepseek/deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are an expert translator. Translate the given Halegannada (Old Kannada) text to Hosa Kannada (Modern Kannada). Provide only the direct translation."},
                    {"role": "user", "content": f"Translate to Hosa Kannada:\n\n{text_to_translate}"}
                ],
                stream=False,
                max_tokens=1000,
                temperature=0.5
            )
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                translated_text = response.choices[0].message.content.strip()
                print("OpenRouter (DeepSeek) Hosa Kannada translation successful.")
                return translated_text
            else:
                print("OpenRouter (DeepSeek) API returned an empty or invalid response for Hosa Kannada translation.")
                return "Hosa Kannada translation not available (empty response)"
        except openai.APIError as e:
            print(f"OpenRouter (DeepSeek) APIError for Hosa Kannada translation: {e.status_code} - {e.message}")
            raise
        except Exception as e:
            print(f"Generic error during OpenRouter (DeepSeek) Hosa Kannada translation: {str(e)}")
            raise

    try:
        return retry_with_backoff(_translate_with_openrouter_deepseek)
    except Exception as e:
        error_message = f"OpenRouter (DeepSeek) Hosa Kannada translation error after retries: {str(e)}"
        print(error_message)
        return "Hosa Kannada translation error: API unavailable or failed"

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/pages/<path:filename>')
def serve_pages(filename):
    return send_from_directory('pages', filename)

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    return send_from_directory('assets', filename)

@app.route('/translate', methods=['POST'])
def translate():
    try:
        print("\n=== New Translation Request ===")
        data = request.json
        
        if not data:
            print("Error: No data received")
            return jsonify({'error': 'No data received', 'status': 'error'}), 400
            
        text_input = data.get('text', '')
        image_data = data.get('image', '')
        source = data.get('source', 'api') # Default to 'api' if source not specified

        english_translation_result = ""
        kannada_translation_result = ""

        if image_data:
            print("Image translation request received")
            # Image processing now returns a dictionary
            processing_result = process_image(image_data)

            if "error" in processing_result:
                error_message = processing_result["error"]
                status_code = 429 if "rate limit" in error_message.lower() else 500
                return jsonify({
                    'error': error_message,
                    'status': 'error',
                    'retry_after': INITIAL_RETRY_DELAY if status_code == 429 else None
                }), status_code

            print(f"Image processing completed. Extracted: '{processing_result.get('extracted_text', '')[:100]}...', Translated: '{processing_result.get('english', '')[:100]}...'")
            return jsonify({
                'extracted_text': processing_result.get('extracted_text', 'No text detected.'),
                'english': processing_result.get('english', 'Translation not available.'),
                'status': 'completed'
            })

        elif text_input:
            print(f"Text translation request received for '{text_input}' with source: {source}")

            # Conditional logic for Hosa Kannada translation
            if source == 'dictionary':
                print("Using dictionary for Kannada translation.")
                kannada_words = text_input.split()
                translated_kannada_words = [get_kannada_translation(word) for word in kannada_words]
                kannada_translation_result = ' '.join(translated_kannada_words)
            else:
                print("Using API for Hosa Kannada translation.")
                kannada_translation_result = get_hosa_kannada_translation_api(text_input)

            # English translation is always done via API for text inputs
            english_translation_result = get_english_translation(text_input)

            print(f"Kannada Translation: {kannada_translation_result}")
            print(f"English Translation: {english_translation_result}")

            return jsonify({
                'kannada': kannada_translation_result,
                'english': english_translation_result,
                'status': 'completed'
            })
        else:
            return jsonify({'error': 'No text or image provided', 'status': 'error'}), 400
    
    except Exception as e:
        error_msg = f"General translation endpoint error: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return jsonify({
            'error': error_msg,
            'status': 'error',
            'retry_after': INITIAL_RETRY_DELAY # Generic retry suggestion for unexpected server errors
        }), 500


@app.route('/translate-halegannada', methods=['POST'])
def translate_halegannada():
    try:
        data = request.json
        text_input = data.get('text', '')
        if not text_input:
            return jsonify({'error': 'No text provided', 'status': 'error'}), 400

        # This endpoint is more specific, let's assume it might have different logic or models in the future.
        # For now, it will also use the dictionary for Kannada and DeepSeek for English.

        print(f"Halegannada translation request received: {text_input}")

        # Get Kannada translation (word by word from dictionary)
        kannada_words = text_input.split()
        translated_kannada_words = [get_kannada_translation(word) for word in kannada_words]
        kannada_translation = ' '.join(translated_kannada_words)

        # Get English translation
        english_translation = get_english_translation(text_input)

        print(f"Kannada Translation: {kannada_translation}")
        print(f"English Translation: {english_translation}")

        return jsonify({
            'kannada_translation': kannada_translation,
            'english_translation': english_translation,
            'status': 'completed'
        })

    except Exception as e:
        error_msg = f"Halegannada translation endpoint error: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return jsonify({'error': error_msg, 'status': 'error'}), 500

if __name__ == '__main__':
    # Ensure the 'assets' directory exists.
    if not os.path.exists('assets'):
        os.makedirs('assets')
        print("Created 'assets' directory as it was missing.")

    # Running with debug=False to prevent the auto-reloader from causing issues during verification.
    # The reloader can be problematic in some environments.
    app.run(debug=False, port=5000)