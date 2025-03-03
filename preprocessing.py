import fitz  # PyMuPDF
import google.generativeai as genai
import os
import dotenv
from PIL import Image
import io

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file using OCR, page by page.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Extracted text as a single string, or None on error.
    """
    dotenv.load_dotenv()
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

    if not GOOGLE_API_KEY:
        print("ERROR: GOOGLE_API_KEY environment variable not set.")
        return None

    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
    except genai.APIError as e:  # Catch specific APIError
        if "invalid API key" in str(e).lower(): # Check for the message
            print("ERROR: Invalid Gemini API key provided.")
        else:
            print(f"ERROR: Gemini API error: {e}") # other api related errors
        return None
    except Exception as e:
        print(f"ERROR: Failed to configure Gemini API: {e}")
        return None

    try:
        doc = fitz.open(pdf_path)
    except FileNotFoundError:
        print(f"ERROR: PDF file not found: {pdf_path}")
        return None
    except fitz.FileDataError:
        print(f"ERROR: Could not open {pdf_path}. It might be corrupted or invalid.")
        return None
    except Exception as e:
        print(f"ERROR opening PDF: {e}")
        return None

    text_by_page = []
    for page_number, page in enumerate(doc):
        try:
            pixmap = page.get_pixmap()
            img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            with io.BytesIO() as img_byte_arr:
                img.save(img_byte_arr, format='PNG')
                response = model.generate_content([
                    {'mime_type': 'image/png', 'data': img_byte_arr.getvalue()},
                    "Extract all the text from this image:"
                ])
            text_by_page.append(response.text)

        except Exception as e:
            print(f"ERROR processing page {page_number + 1}: {e}")
            return None  # Stop on page error

    doc.close()
    return "".join(text_by_page)

if __name__ == '__main__':
    pdf_file = "The_Gift_of_the_Magi.pdf"
    if os.path.exists(pdf_file):
      extracted_text = extract_text_from_pdf(pdf_file)
      if extracted_text:
          print(extracted_text)
    else:
      print(f"Error: PDF file '{pdf_file}' not found.")
