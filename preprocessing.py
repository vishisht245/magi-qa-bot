import fitz  # PyMuPDF
import google.generativeai as genai
import os
import dotenv
from PIL import Image
import io

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file using OCR, page by page,
    and returns the combined text as a single string.

    Args:
        pdf_path: The path to the PDF file.

    Returns:
        A single string containing all the extracted text.
    """
    dotenv.load_dotenv()
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)

    model = genai.GenerativeModel('gemini-1.5-flash')

    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_number, page in enumerate(doc):
        pixmap = page.get_pixmap()
        img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples) 
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG') # Convert pdf to png 
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

        print(f"--- Page {page_number + 1} ---")
        print(f"Image bytes length: {len(img_byte_arr)}")
        # with open(f"page_{page_number + 1}.png", "wb") as f: # Debugging
        #     f.write(img_byte_arr)

        text_by_page.append(response.text)

    doc.close()
    return "".join(text_by_page)


if __name__ == '__main__':
    pdf_file = "The_Gift_of_the_Magi.pdf"
    extracted_text = extract_text_from_pdf(pdf_file)
    print(extracted_text)
