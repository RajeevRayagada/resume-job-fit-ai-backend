from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai import Agent

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

import logging
import json

# --------------------
# Setup
# --------------------

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Resume Job Fit AI")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://resume-job-fit-ai-frontend.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------
# Models
# --------------------

class ResumeRequest(BaseModel):
    resume_text: str = Field(
        ...,
        min_length=50,
        description="Resume text must be at least 50 characters"
    )
    job_description: str = Field(
        ...,
        min_length=50,
        description="Job description must be at least 50 characters"
    )

class ResumeResponse(BaseModel):
    match_score: int
    strengths: list[str]
    missing_skills: list[str]
    suggestions: list[str]

# --------------------
# AI Agent
# --------------------

agent = Agent(
    model=OpenRouterModel(
        model_name="openai/gpt-4o-mini"
    ),
    system_prompt=(
        "You are a professional resume evaluator.\n"
        "Analyze the resume against the job description.\n"
        "Return ONLY valid JSON with the following fields:\n"
        "match_score (integer 0-100),\n"
        "strengths (array of strings),\n"
        "missing_skills (array of strings),\n"
        "suggestions (array of strings).\n"
        "Do not include explanations or extra text."
    ),
)


# --------------------
# Endpoint
# --------------------

@app.post("/analyze", response_model=ResumeResponse)
async def analyze_resume(data: ResumeRequest):
    try:
        result = await agent.run(
            f"""
Resume:
{data.resume_text}

Job Description:
{data.job_description}
"""
        )

        # Parse raw model output
        parsed = json.loads(result.output)

        # Validate structure
        validated = ResumeResponse(**parsed)

        return validated

    except (json.JSONDecodeError, ValidationError):
        logger.exception("Model returned invalid JSON")
        raise HTTPException(
            status_code=502,
            detail="AI returned an invalid response. Please try again."
        )

    except Exception:
        logger.exception("AI agent failed")
        raise HTTPException(
            status_code=502,
            detail="AI service is temporarily unavailable. Please try again."
        )
