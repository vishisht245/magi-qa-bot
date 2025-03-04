import fitz
import google.generativeai as genai
import os
import dotenv
from PIL import Image
import io

def extract_text_from_pdf(pdf_input):
    """
    Extracts text from a PDF (file path or BytesIO) using OCR, page by page.
    Returns the combined text as a single string.
    """
    dotenv.load_dotenv()
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)

    model = genai.GenerativeModel('gemini-1.5-flash')

    if isinstance(pdf_input, str):
        doc = fitz.open(pdf_input)
    elif isinstance(pdf_input, io.BytesIO):
        doc = fitz.open("pdf", pdf_input.read())
    else:
        raise ValueError("pdf_input must be a file path (string) or a BytesIO object.")

    text_by_page = []

    for page_number, page in enumerate(doc):
        pixmap = page.get_pixmap()
        img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples) # Convert pdf to image
        img_byte_arr = io.BytesIO() # Saving in-memory
        img.save(img_byte_arr, format='PNG') # Saving in png format
        img_byte_arr = img_byte_arr.getvalue()

        prompt = "Extract all the text from this image:"

        response = model.generate_content(
            [
                {
                    'mime_type': 'image/png',
                    'data': img_byte_arr
                },
                prompt,
            ]
        )

        # Debug prints 
        print(f"--- Page {page_number + 1} ---")
        print(f"Image bytes length: {len(img_byte_arr)}")
        # Debugging: save images
        # with open(f"page_{page_number + 1}.png", "wb") as f:
        #     f.write(img_byte_arr)

        text_by_page.append(response.text)

    doc.close()
    return "".join(text_by_page)

if __name__ == '__main__':
    pdf_file = "The_Gift_of_the_Magi.pdf"
    extracted_text = extract_text_from_pdf(pdf_file)
    print(extracted_text)
