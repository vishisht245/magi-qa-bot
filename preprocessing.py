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

    Args:
        pdf_input: File path (string) OR a BytesIO object.

    Returns:
        A single string containing all the extracted text.
    """
    dotenv.load_dotenv()  
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") 
    genai.configure(api_key=GOOGLE_API_KEY)  

    model = genai.GenerativeModel('gemini-1.5-flash')

    # Handle both file paths and BytesIO objects (for Streamlit uploads)
    if isinstance(pdf_input, str):
        doc = fitz.open(pdf_input)  
    elif isinstance(pdf_input, io.BytesIO):
        doc = fitz.open("pdf", pdf_input.read()) 
    else:
        raise ValueError("pdf_input must be a file path (string) or a BytesIO object.")

    text_by_page = []  # Store extracted text from each page

    for page_number, page in enumerate(doc):  
        pixmap = page.get_pixmap()  # Get the page as an image 
        img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)  # Create PIL Image from pixmap
        img_byte_arr = io.BytesIO()  
        img.save(img_byte_arr, format='PNG')  # Save the pil Image as PNG 
        img_byte_arr = img_byte_arr.getvalue()  # Get the image data as bytes

        prompt = "Extract all the text from this image:"

        # Send image and prompt to Gemini for OCR, including mime type
        response = model.generate_content(
            [
                {
                    'mime_type': 'image/png',  # Specify MIME type for the image
                    'data': img_byte_arr       # Image data as bytes
                },
                prompt,  
            ]
        )

        print(f"--- Page {page_number + 1} ---") # Debug Print
        print(f"Image bytes length: {len(img_byte_arr)}")
        # with open(f"page_{page_number + 1}.png", "wb") as f: # Debugging: save images
        #     f.write(img_byte_arr)

        text_by_page.append(response.text) 

    doc.close()  
    return "".join(text_by_page) # Combine all page text and return


if __name__ == '__main__':
    #  for testing
    pdf_file = "The_Gift_of_the_Magi.pdf"  
    extracted_text = extract_text_from_pdf(pdf_file)
    print(extracted_text)
