import pdfplumber
from openai import OpenAI
import json
import os

client = OpenAI(api_key="key")
# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            full_text += page.extract_text()
    return full_text

def extract_experience_section(text):
    system_prompt = f"""
    You are a highly skilled assistant. 
    Your task is to extract job experience information from a CV PDFs and return it in a very specific JSON format. 
    Please ensure that the output follows this format exactly and returns no extra text, explanations, or characters.
       You are an OCR-like data extraction tool that extracts hotel invoice data from PDFs.

       1. Please extract the data of experience, grouping data according to theme/sub groups, and then output into JSON.

       2. Please keep the keys and values of the JSON in the original language. 

       3. The type of data you might encounter in the invoice includes but is not limited to: dates, company and position. 

       4. If the page contains no charge data, please output an empty JSON object and don't make up any data.

       5. If there are blank data fields in the invoice, please include them as "Not found" values in the JSON object.

       6. Don't interpolate or make up data.
       """
    # Define the system and user prompts
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"""Extract job experiences from this CV text in the following JSON Object format. If you don't found any information write Not found. Only return the JSON array without any extra text or \\n characters:
            [
                {{
                    "dates": "[MM/YYYY - MM/YYYY]", 
                    "company": "Company Name", 
                    "position": "Job Title"
                }}
            ]
            **Important**: Use **[MM/YYYY - MM/YYYY]** format for dates with MM as strictly **2-digit numeric** (e.g., 01 for January, 11 for November), with no text-based month names.
            **Important**: Respect the following JSON format **without any extra text or \\n characters**.
            Here is the CV text:{text}"""
         }
    ]

    # Call the OpenAI API with the new ChatCompletion method
    response = client.chat.completions.create(model="gpt-4-turbo",  # or "gpt-4, gpt-3.5-turbo, gpt-4o und gpt-4o-mini"
                                              messages=messages,
                                              temperature=0.3,
                                              max_tokens=1500,
                                              n=1)
    return response.choices[0].message.content.strip()
def main(pdf_directory, output_json_path):
    all_experience_data = {}

    # Loop through all PDF files in the specified directory
    for pdf_filename in os.listdir(pdf_directory):
        if pdf_filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, pdf_filename)
            full_text = extract_text_from_pdf(pdf_path)
            experience_details = extract_experience_section(full_text)
            try:
                # Attempt to parse the JSON response
                all_experience_data[pdf_filename] = json.loads(experience_details)
            except json.JSONDecodeError:
                # If there's an error, only store the `output` content for this PDF
                cleaned_output = experience_details.strip("```json\n").rstrip("```")  # Remove the markdown formatting
                all_experience_data[pdf_filename] = cleaned_output  # Store only the output text
    # Write all extracted data to a single JSON file
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(all_experience_data, json_file, ensure_ascii=False, indent=4)
    return all_experience_data
pdf_directory = "cv_30"
output_json_path = "extracted_data_gpt_4_turbo.json"
experience_data = main(pdf_directory, output_json_path)
# Display the output
print(json.dumps(experience_data, indent=4))
