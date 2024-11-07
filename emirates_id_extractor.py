import boto3
import json
from typing import Dict, Optional
import os
import re
from datetime import datetime
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

class EmiratesIDExtractor:
    def __init__(self, region_name: str, aws_access_key_id: str, aws_secret_access_key: str, openai_api_key: str):
        """Initialize the Emirates ID Extractor with AWS and OpenAI credentials."""
        self.textract_client = boto3.client(
            "textract",
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )

        self.s3_client = boto3.client(
            "s3",
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )

        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=32,
            length_function=len,
        )
        self.embeddings = OpenAIEmbeddings()
        self.llm = OpenAI(temperature=0)

    def process_and_query(self, text: str) -> Dict[str, str]:
        """Process extracted text using LangChain and OpenAI."""
        try:
            # Updated regex patterns to better match the exact format
            patterns = {
                'name': r'Name\s*(?:[:\n]|\s{2,})\s*([A-Z\s]+(?=[^a-z]|$))',
                'id_number': r'(?:ID Number|U\.I\.D\.No)\s*(?:[:\n]|\s{2,})\s*([A-Z0-9\s/-]+)(?=[^a-z0-9]|$)',
                'nationality': r'Nationality\s*(?:[:\n]|\s{2,})\s*([A-Z\s]+(?=[^a-z]|$))',
                'passport_no': r'Passport\s*(?:No|Number)\s*(?:[:\n]|\s{2,})\s*([A-Z0-9]+)(?=[^a-z0-9]|$)',
                'profession': r'Profession\s*(?:[:\n]|\s{2,})\s*([A-Za-z\s()]+)(?=[^a-z]|$)',
                'sponsor': r'Sponsor\s*(?:[:\n]|\s{2,})\s*([A-Z\s&.]+(?=[^a-z]|$))',
                'place_of_issue': r'Place\s*(?:Of|of)\s*Issue\s*(?:[:\n]|\s{2,})\s*([A-Za-z\s]+)(?=[^a-z]|$)',
                'issue_date': r'Issue\s*Date\s*(?:[:\n]|\s{2,})\s*(\d{4}/\d{2}/\d{2})',
                'expiry_date': r'Expiry\s*Date\s*(?:[:\n]|\s{2,})\s*(\d{4}/\d{2}/\d{2})'
            }

            extracted_info = {}
            
            # Process only lines that contain English text
            english_lines = []
            for line in text.split('\n'):
                # Check if line contains English text and no Arabic
                if re.search(r'[A-Za-z]', line) and not re.search(r'[\u0600-\u06FF]', line):
                    english_lines.append(line)
            
            english_text = '\n'.join(english_lines)

            # Extract information using patterns
            for field, pattern in patterns.items():
                match = re.search(pattern, english_text, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    if value and value.lower() != 'not found':
                        extracted_info[field] = value

            return extracted_info

        except Exception as e:
            raise Exception(f"Error in text processing: {str(e)}")

    def extract_text_from_image(self, image_path: str, bucket_name: str) -> Dict[str, str]:
        """Extract text from Emirates ID image using Amazon Textract and process with LLM."""
        try:
            s3_file_path = f"emirates_ids/{os.path.basename(image_path)}"
            s3_uri = self.upload_to_s3(image_path, bucket_name, s3_file_path)

            response = self.textract_client.detect_document_text(
                Document={
                    'S3Object': {
                        'Bucket': bucket_name,
                        'Name': s3_file_path
                    }
                }
            )

            # Extract text, focusing on English lines
            extracted_lines = []
            for block in response['Blocks']:
                if block['BlockType'] == 'LINE':
                    text = block['Text']
                    # Only include if contains English and no Arabic
                    if re.search(r'[A-Za-z]', text) and not re.search(r'[\u0600-\u06FF]', text):
                        extracted_lines.append(text)

            extracted_text = "\n".join(extracted_lines)
            extracted_info = self.process_and_query(extracted_text)

            self.s3_client.delete_object(Bucket=bucket_name, Key=s3_file_path)
            return extracted_info

        except Exception as e:
            raise Exception(f"Error processing Emirates ID: {str(e)}")

    def upload_to_s3(self, local_file_path: str, bucket_name: str, s3_file_path: str) -> str:
        """Upload the image to S3 bucket."""
        try:
            self.s3_client.upload_file(local_file_path, bucket_name, s3_file_path)
            return f"s3://{bucket_name}/{s3_file_path}"
        except Exception as e:
            raise Exception(f"Error uploading to S3: {str(e)}")