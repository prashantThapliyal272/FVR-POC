from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from openai import OpenAI
from dotenv import load_dotenv
import json
import os
# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()


def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    doc = DocumentFile.from_pdf(file_bytes)
    model = ocr_predictor(pretrained=True)
    result = model(doc)

    full_text = ""
    for page in result.export()["pages"]:
        for block in page["blocks"]:
            for line in block["lines"]:
                words = [word["value"] for word in line["words"]]
                full_text += " ".join(words) + "\n"
    return full_text



def get_structured_data_from_text(text):
    system_prompt = (
        "You are an AI that extracts structured fields from invoices or logistics documents.\n"
        "Format the output like this JSON example:\n"
        "{\n"
        '  "Company Name": "KWE Kintetsu World Express (Taiwan), Inc.",\n'
        '  "Company Address": {\n'
        '    "Address 1": "3FL.NO.99.SEC.2.CHANG. AN El ROAD",\n'
        '    "Address 2": "ROOM 3,9TH FL.NO.412.CHUNG-SHANG",\n'
        '    "City": "TAIPEI",\n'
        '    "Postal Code": "104",\n'
        '    "Country": "TAIWAN"\n'
        '  },\n'
        '  "Second Address": "2ND RD.KAOHSIUNG 802,TAIWAN",\n'
        '  "Telephone Numbers": ["(02)2506-3151", "(07)332-0907", "(07)332-0037"],\n'
        '  "Fax Numbers": ["(02)2506-5735", "(07)332-0913"],\n'
        '  "Invoice Type": "TAX INVOICE",\n'
        '  "Page Number": "1/1",\n'
        '  "Invoice Category": "AIR IMPORT",\n'
        '  "GST Registration Number": "",\n'
        '  "Bill To": "STRYKER FAR EAST INC. TAIWAN BRANCH (EDI BILL\' TO ONLY)",\n'
        '  "Invoice Number": "59302142414300",\n'
        '  "Bill To Address": "5F, 1 NO. 100, SEC. 2, ROOSEVELT RD, TAIPEI 10084",\n'
        '  "Invoice Date": "11-12-24",\n'
        '  "Reference Number": "302264514",\n'
        '  "Payment Terms": "30 NET",\n'
        '  "Master Airway Bill Number": "297-60287673",\n'
        '  "House Airway Bill Number": "330014937785",\n'
        '  "Flight Number and Date": "CI 5231 / Nov.18, 2024",\n'
        '  "Pieces": 5,\n'
        '  "Weight": 1379.0,\n'
        '  "Origin/Destination": "ORD/ TPE",\n'
        '  "Description and Remarks": "MEDICAL SPARE PARTS",\n'
        '  "Charges": {\n'
        '    "Delivery Charge": 8511.00,\n'
        '    "Heavy Lift Surcharge": 552.00,\n'
        '    "Duty & Tax": 38249.00,\n'
        '    "Handling": 945.00,\n'
        '    "Customs Entry Fee (Formal)": 1260.00,\n'
        '    "Terminal Service Fee Destination": 13439.00,\n'
        '    "Airfreight-Inbound": 63842.00,\n'
        '    "GST Output Tax": 1235.00,\n'
        '    "Standard Rated Amount": 24707.00,\n'
        '    "Zero Rated Amount": 102091.00\n'
        '  },\n'
        '  "Total Amount": 128033.00,\n'
        '  "Payment Method": "Cross Cheque payable to Kinetsul Express (Taiwan), Inc. within Credit Term",\n'
        '  "Invoice Requirement": "This Computer Generated Invoice Requires No Signature",\n'
        '  "Invoice Status": "ORIGINAL INVOICE",\n'
        '  "Errors and Omissions Excepted": "E.& O.E."\n'
        '}'
    )

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract structured JSON data from the following OCR text:\n\n{text}"}
        ],
        temperature=0.1
    )
    return response.choices[0].message.content


@app.get("/")
def health_check():
    return {"status": "API is alive"}




@app.post("/extract-binary")
async def extract_from_binary(request: Request):
    try:
        file_bytes = await request.body()
        extracted_text = extract_text_from_pdf_bytes(file_bytes)
        structured_data = get_structured_data_from_text(extracted_text)
        return JSONResponse(content=structured_data)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
