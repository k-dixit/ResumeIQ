import json
import logging
import os
import time

import azure.functions as func
import requests
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.storage.blob import BlobServiceClient
from openai import OpenAI

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# config
BLOB_URL = os.getenv("BLOB_URL")
DOC_INTEL_ENDPOINT = os.getenv("DOC_INTEL_ENDPOINT")
DOC_INTEL_KEY = os.getenv("DOC_INTEL_KEY")
FOUNDRY_PROJECT_ENDPOINT = os.getenv("FOUNDRY_PROJECT_ENDPOINT")
CHAT_MODEL_DEPLOYMENT_NAME = os.getenv(
    "CHAT_MODEL_DEPLOYMENT_NAME",
    "gpt-4.1-mini"
)

# validation
_required_env = {
    "BLOB_URL": BLOB_URL,
    "DOC_INTEL_ENDPOINT": DOC_INTEL_ENDPOINT,
    "FOUNDRY_PROJECT_ENDPOINT": FOUNDRY_PROJECT_ENDPOINT,
}
for _name, _value in _required_env.items():
    if not _value:
        raise ValueError(f"Required environment variable '{_name}' is not set.")

# auth
credential = DefaultAzureCredential(
    exclude_interactive_browser_credential=True
)

blob_service = BlobServiceClient(
    account_url=BLOB_URL,
    credential=credential
)

token_provider = get_bearer_token_provider(
    credential,
    "https://ai.azure.com/.default"
)

openai_client = OpenAI(
    base_url=FOUNDRY_PROJECT_ENDPOINT.rstrip("/") + "/openai/v1",
    api_key=token_provider
)

# constants
DOC_INTEL_API_VERSION = "2024-11-30"
POLL_INTERVAL_SECONDS = 1
MAX_POLL_ATTEMPTS = 120       # 2 minutes max before timeout


# helpers

def extract_text_from_blob(container_name: str, blob_name: str) -> str:
    """Download a blob and extract its text via Document Intelligence."""
    container = blob_service.get_container_client(container_name)
    blob_data = container.download_blob(blob_name).readall()

    analyze_url = (
        f"{DOC_INTEL_ENDPOINT}"
        "/documentintelligence/documentModels/prebuilt-read:analyze"
        f"?api-version={DOC_INTEL_API_VERSION}"
    )
    headers = {
        "Ocp-Apim-Subscription-Key": DOC_INTEL_KEY,
        "Content-Type": "application/octet-stream",
    }

    response = requests.post(analyze_url, data=blob_data, headers=headers, timeout=60)
    response.raise_for_status()

    operation_location = response.headers.get("Operation-Location")
    if not operation_location:
        raise ValueError("No 'Operation-Location' header in Document Intelligence response.")

    poll_headers = {"Ocp-Apim-Subscription-Key": DOC_INTEL_KEY}

    for attempt in range(MAX_POLL_ATTEMPTS):
        poll_response = requests.get(operation_location, headers=poll_headers, timeout=30)
        poll_response.raise_for_status()
        result = poll_response.json()

        status = result.get("status")
        if status == "succeeded":
            break
        if status == "failed":
            raise ValueError(f"Document Intelligence analysis failed: {result.get('error')}")

        logger.debug("Poll attempt %d: status=%s", attempt + 1, status)
        time.sleep(POLL_INTERVAL_SECONDS)
    else:
        raise TimeoutError(
            f"Document Intelligence did not complete within "
            f"{MAX_POLL_ATTEMPTS * POLL_INTERVAL_SECONDS} seconds."
        )

    lines = [
        line.get("content", "")
        for page in result.get("analyzeResult", {}).get("pages", [])
        for line in page.get("lines", [])
    ]
    return "\n".join(lines)


def save_text_to_blob(container_name: str, blob_name: str, content: str) -> None:
    """Upload a UTF-8 string to Blob Storage, overwriting if it already exists."""
    container = blob_service.get_container_client(container_name)
    container.upload_blob(name=blob_name, data=content.encode("utf-8"), overwrite=True)


def _empty_score_result() -> dict:
    return {
        "score": 0,
        "breakdown": {},
        "matched_keywords": [],
        "missing_keywords": [],
    }


def _strip_json_fences(text: str) -> str:
    """Remove Markdown code fences (``` or ```json) from a string."""
    text = text.strip()
    if text.startswith("```"):
        text = text[text.index("\n") + 1:] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def call_openai_for_scoring(jd_text: str, resume_text: str) -> dict:
    """Score the resume against the job description using Foundry GPT."""
    prompt = f"""Analyze this resume against the job description and provide an ATS compatibility score.

JOB DESCRIPTION:
{jd_text}

RESUME:
{resume_text}

SCORING METHODOLOGY (0-100 scale):
1. Keyword Match (0-40 points): Count exact and synonym matches of required skills, technologies, and qualifications
2. Experience Alignment (0-30 points): Years of experience, relevant roles, and industry match
3. Skills Coverage (0-30 points): Depth and breadth of technical/soft skills matching JD requirements

INSTRUCTIONS:
- Be strict and objective in scoring
- Only count clear, verifiable matches
- Missing critical requirements should significantly lower the score
- Return ONLY valid JSON with this exact structure:
{{
  "score": <total 0-100>,
  "breakdown": {{
    "keyword_match": <0-40>,
    "experience_alignment": <0-30>,
    "skills_coverage": <0-30>
  }},
  "matched_keywords": ["keyword1", "keyword2"],
  "missing_keywords": ["missing1", "missing2"]
}}
"""

    response = openai_client.chat.completions.create(
    model=CHAT_MODEL_DEPLOYMENT_NAME,
    messages=[
        {
            "role": "system",
            "content": (
                "You are an ATS resume analyzer. "
                "Analyze the resume against the job description. "
                "Return ONLY valid JSON. Do not return markdown, "
                "code fences, explanations, or the word null."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ],
    response_format={"type": "json_object"},
    temperature=0.2
)

    raw_content = (response.choices[0].message.content or "").strip()

    if not raw_content:
        logger.error("Empty response received from OpenAI chat model.")
        return _empty_score_result()

    cleaned = _strip_json_fences(raw_content)

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.error("Failed to parse JSON from model response: %s", cleaned[:300])
        return _empty_score_result()
    return result


# azure function app

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


@app.route(route="ui", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def index(req: func.HttpRequest) -> func.HttpResponse:
    """Serve the HTML UI for the resume screening interface."""
    try:
        with open("index.html", encoding="utf-8") as f:
            html_content = f.read()
        return func.HttpResponse(html_content, mimetype="text/html", status_code=200)
    except FileNotFoundError:
        return func.HttpResponse(
            "index.html not found. Please ensure the file is deployed with the function.",
            status_code=404,
        )


@app.route(route="upload_resume", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
def upload_resume(req: func.HttpRequest) -> func.HttpResponse:
    """Upload resume file to Blob Storage (multipart form upload from the frontend)."""
    try:
        files = req.files
        if not files or "file" not in files:
            return func.HttpResponse(
                json.dumps({"error": "No file uploaded"}),
                status_code=400,
                mimetype="application/json",
            )

        file = files["file"]
        filename = file.filename
        file_content = file.read()

        container = blob_service.get_container_client("original-resume")
        container.upload_blob(name=filename, data=file_content, overwrite=True)

        return func.HttpResponse(
            json.dumps({"blob_path": f"original-resume/{filename}"}),
            status_code=200,
            mimetype="application/json",
        )
    except Exception:
        logger.exception("Resume upload failed.")
        return func.HttpResponse(
            json.dumps({"error": "Upload failed. Please try again."}),
            status_code=500,
            mimetype="application/json",
        )


@app.route(route="optimize_resume", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
def main(req: func.HttpRequest) -> func.HttpResponse:
    """Extract resume text, score it against the job description, and save the report."""
    try:
        payload = req.get_json()
    except ValueError:
        return func.HttpResponse(
            json.dumps({"error": "Request body must be valid JSON."}),
            status_code=400,
            mimetype="application/json",
        )

    resume_blob = (payload.get("resume_blob") or "").strip()
    jd_text = (payload.get("job_description") or "").strip()

    if not resume_blob or not jd_text:
        return func.HttpResponse(
            json.dumps({"error": "'resume_blob' and 'job_description' are required."}),
            status_code=400,
            mimetype="application/json",
        )

    if "/" not in resume_blob:
        return func.HttpResponse(
            json.dumps({"error": "'resume_blob' must be in 'container/blob' format."}),
            status_code=400,
            mimetype="application/json",
        )

    container_name, blob_name = resume_blob.split("/", 1)

    try:
        raw_text = extract_text_from_blob(container_name, blob_name)
        score_result = call_openai_for_scoring(jd_text, raw_text)

        base_name = blob_name.rsplit(".", 1)[0] if "." in blob_name else blob_name
        save_text_to_blob(
            "resume-reports",
            f"{base_name}_report.json",
            json.dumps({"score": score_result}),
        )

        return func.HttpResponse(
            json.dumps(score_result),
            status_code=200,
            mimetype="application/json",
        )
    except Exception:
        logger.exception("Resume scoring failed.")
        return func.HttpResponse(
            json.dumps({"error": "Scoring failed. Please try again."}),
            status_code=500,
            mimetype="application/json",
        )