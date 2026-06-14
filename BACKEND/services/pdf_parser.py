import io
from pypdf import PdfReader

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extracts all text from a PDF file provided as bytes.
    """
    text = ""
    try:
        # Open the PDF from bytes
        reader = PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    except Exception as e:
        raise ValueError(f"Failed to parse PDF: {str(e)}")
        
    return text.strip()
