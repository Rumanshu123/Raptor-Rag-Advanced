import os
import streamlit as st
import traceback
from pypdf import PdfReader

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
        return text if text.strip() else "No text could be extracted."
    except Exception as e:
        st.error(f"Error extracting PDF: {str(e)}")
        st.code(traceback.format_exc())
        return ""
