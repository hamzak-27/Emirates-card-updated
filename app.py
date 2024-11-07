import streamlit as st
import json
import base64
from emirates_id_extractor import EmiratesIDExtractor
import tempfile
import os

# Set page config
st.set_page_config(
    page_title="Emirates ID Information Extractor",
    page_icon="🆔",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .stApp {
        background-color: #1a1a1a;
    }
    
    .description {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: #ffffff90;
    }
    
    .stButton > button {
        width: 100%;
        background-color: #0e76a8 !important;
        color: white !important;
        transition: background-color 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background-color: #0a5a8a !important;
        border-color: #0a5a8a !important;
    }
    
    .field-label {
        color: rgba(255, 255, 255, 0.6);
        font-size: 14px;
        margin-bottom: 5px;
    }
    
    .field-value {
        color: white;
        font-size: 14px;
        margin-bottom: 15px;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

def display_results(col1, col2, extracted_info):
    """Display the extracted information in two columns."""
    # Priority fields for first column
    priority_fields = {
        'name': 'Name',
        'id_number': 'ID Number',
        'nationality': 'Nationality',
        'passport_no': 'Passport Number'
    }
    
    # Secondary fields for second column
    secondary_fields = {
        'profession': 'Profession',
        'sponsor': 'Sponsor',
        'place_of_issue': 'Place of Issue',
        'issue_date': 'Issue Date',
        'expiry_date': 'Expiry Date'
    }
    
    # Display priority fields in first column
    with col1:
        for key, label in priority_fields.items():
            if key in extracted_info:
                st.markdown(f'<div class="field-label">{label}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="field-value">{extracted_info[key]}</div>', unsafe_allow_html=True)
    
    # Display secondary fields in second column
    with col2:
        for key, label in secondary_fields.items():
            if key in extracted_info:
                st.markdown(f'<div class="field-label">{label}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="field-value">{extracted_info[key]}</div>', unsafe_allow_html=True)

def main():
    st.title("Emirates ID Information Extractor")
    
    st.markdown("""
        <div class="description">
            This tool extracts information from Emirates ID cards using advanced AI processing. 
            Upload your Emirates ID image and click 'Process the Card' to get the extracted information.
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize credentials from secrets
    AWS_REGION = st.secrets["aws"]["region"]
    AWS_ACCESS_KEY = st.secrets["aws"]["access_key"]
    AWS_SECRET_KEY = st.secrets["aws"]["secret_key"]
    OPENAI_API_KEY = st.secrets["openai"]["api_key"]
    BUCKET_NAME = st.secrets["aws"]["bucket_name"]

    @st.cache_resource
    def get_extractor():
        return EmiratesIDExtractor(
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            openai_api_key=OPENAI_API_KEY
        )

    extractor = get_extractor()

    if 'current_file_name' not in st.session_state:
        st.session_state.current_file_name = None

    uploaded_file = st.file_uploader("Upload Emirates ID", type=['jpg', 'jpeg', 'png', 'pdf'])

    if uploaded_file is not None:
        current_file_name = uploaded_file.name
        if (st.session_state.current_file_name != current_file_name):
            st.session_state.current_file_name = current_file_name
            if 'extracted_info' in st.session_state:
                del st.session_state.extracted_info
            if 'file_content' in st.session_state:
                del st.session_state.file_content
        
        if 'file_content' not in st.session_state:
            st.session_state.file_content = uploaded_file.read()

        if st.button("Process the Card", type="primary"):
            with st.spinner("Processing Emirates ID..."):
                try:
                    file_extension = os.path.splitext(uploaded_file.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                        tmp_file.write(st.session_state.file_content)
                        temp_path = tmp_file.name

                    extracted_info = extractor.extract_text_from_image(temp_path, BUCKET_NAME)
                    os.unlink(temp_path)

                    st.session_state.extracted_info = extracted_info

                except Exception as e:
                    st.error(f"Error processing Emirates ID: {str(e)}")

        if hasattr(st.session_state, 'extracted_info') and st.session_state.extracted_info:
            col1, col2 = st.columns(2)
            display_results(col1, col2, st.session_state.extracted_info)

if __name__ == "__main__":
    main()

#updated one