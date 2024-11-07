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

    def upload_to_s3(self, local_file_path: str, bucket_name: str, s3_file_path: str) -> str:
        """Upload the image to S3 bucket."""
        try:
            self.s3_client.upload_file(local_file_path, bucket_name, s3_file_path)
            return f"s3://{bucket_name}/{s3_file_path}"
        except Exception as e:
            raise Exception(f"Error uploading to S3: {str(e)}")

    def process_and_query(self, text: str) -> Dict[str, str]:
        """Process extracted text using LangChain and OpenAI."""
        try:
            texts = self.text_splitter.split_text(text)
            docsearch = FAISS.from_texts(texts, self.embeddings)
            chain = load_qa_chain(self.llm, chain_type="stuff")
            
            # Updated prompt to focus on English text and include nationality
            query = """Extract the following details from the text. Focus ONLY on English text parts, completely ignore any Arabic text.
            
            1. Name (only English name, ignore Arabic name)
            2. ID Number (numbers after 'ID Number' in English)
            3. Nationality (only English text after 'Nationality:')
            4. Passport No. (if available)
            5. Profession (if available)
            6. Sponsor (if available)
            7. Place of Issue (if available)
            8. Issue Date (if available, in format YYYY/MM/DD)
            9. Expiry Date (if available, in format YYYY/MM/DD)

            Important Rules:
            - Strictly extract from English text only
            - For ID Number, include the full number including dashes
            - For Name, only use the English version, ignore Arabic text
            - For Nationality, only use the English version
            - Only include fields that are actually present in the text
            - Skip any fields that aren't found rather than marking as 'Not Found'
            
            Format the response as a JSON object with only the fields that are found:
            {
                "name": "if found",
                "id_number": "if found",
                "nationality": "if found",
                ... (other fields only if found)
            }
            
            Only return the JSON object with found fields, nothing else."""
            
            docs = docsearch.similarity_search(query)
            result = chain.run(input_documents=docs, question=query)
            
            try:
                extracted_info = json.loads(result)
                # Clean up any "Not Found" values
                return {k: v for k, v in extracted_info.items() if v != "Not Found"}
                        
            except json.JSONDecodeError:
                extracted_info = self.extract_using_regex(result)
                return {k: v for k, v in extracted_info.items() if v != "Not Found"}
            
        except Exception as e:
            raise Exception(f"Error in LLM processing: {str(e)}")

    def extract_using_regex(self, text: str) -> Dict[str, str]:
        """Fallback method to extract information using regex patterns."""
        extracted_info = {}
        
        # Extract only what we find
        # Name pattern (English only)
        name_match = re.search(r'Name:\s*([A-Za-z\s]+)', text)
        if name_match:
            extracted_info["name"] = name_match.group(1).strip()
        
        # ID Number pattern
        id_match = re.search(r'ID Number[:/\s]+([0-9-]+)', text)
        if id_match:
            extracted_info["id_number"] = id_match.group(1).strip()
            
        # Nationality pattern (English only)
        nationality_match = re.search(r'Nationality:\s*([A-Za-z\s]+)', text)
        if nationality_match:
            extracted_info["nationality"] = nationality_match.group(1).strip()
        
        # Only add other fields if found
        passport_match = re.search(r'Z\d{7}', text)
        if passport_match:
            extracted_info["passport_no"] = passport_match.group(0)
        
        profession_match = re.search(r'Profession:\s*([A-Za-z\s]+)', text)
        if profession_match:
            extracted_info["profession"] = profession_match.group(1).strip()
        
        place_match = re.search(r'(Dubai|Abu Dhabi|Sharjah|Ajman|Umm Al Quwain|Ras Al Khaimah|Fujairah)',
                              text, re.IGNORECASE)
        if place_match:
            extracted_info["place_of_issue"] = place_match.group(1)
        
        return extracted_info

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

            extracted_text = " ".join([
                block['Text'] for block in response['Blocks']
                if block['BlockType'] == 'LINE'
            ])

            extracted_info = self.process_and_query(extracted_text)

            self.s3_client.delete_object(Bucket=bucket_name, Key=s3_file_path)

            return extracted_info

        except Exception as e:
            raise Exception(f"Error processing Emirates ID: {str(e)}")