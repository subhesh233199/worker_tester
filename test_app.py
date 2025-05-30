import os
import base64
import hashlib
import json
import logging
import sqlite3
import time
import shutil
from typing import Dict, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from state import shared_state
from TM_crew_setup import setup_crew, validate_metrics, run_fallback_visualization
from pdf_extractor_from_folder import get_pdf_files_from_folder
from TM_pdf_extractor import extract_text_from_pdfs
from TM_enhance_markdown import enhance_report_markdown
from report_validation import validate_report
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    folder_path: str

    @validator('folder_path')
    def validate_folder_path(cls, v):
        if not v:
            raise ValueError('Folder path cannot be empty')
        return v

class AnalysisResponse(BaseModel):
    metrics: Dict
    visualizations: List[str]
    report: str
    evaluation: Dict
    hyperlinks: List[Dict]

class SaveReportRequest(BaseModel):
    folder_path: str
    report: str
    metrics: Dict

    @validator('folder_path')
    def validate_folder_path(cls, v):
        if not v:
            raise ValueError('Folder path cannot be empty')
        return v

    @validator('metrics')
    def validate_metrics_data(cls, v):
        if not validate_metrics(v):
            raise ValueError('Invalid metrics data structure')
        return v

    @validator('report')
    def validate_report_content(cls, v):
        if not validate_report(v):
            raise ValueError('Report missing required sections')
        return v

def hash_string(s: str) -> str:
    """Generate SHA-256 hash of a string."""
    return hashlib.sha256(s.encode()).hexdigest()

def get_base64_image(image_path: str) -> str:
    """Convert an image file to base64 string."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {str(e)}")
        return ""

def convert_windows_path(folder_path: str) -> str:
    """Convert Windows path to Unix-style path."""
    return folder_path.replace('\\', '/')

def hash_pdf_contents(pdf_files: List[str]) -> str:
    """Generate SHA-256 hash of contents of PDF files."""
    hasher = hashlib.sha256()
    for pdf_file in sorted(pdf_files):
        try:
            with open(pdf_file, 'rb') as f:
                hasher.update(f.read())
        except Exception as e:
            logger.error(f"Failed to hash PDF {pdf_file}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to hash PDF {pdf_file}")
    return hasher.hexdigest()

def get_cached_report(folder_path_hash: str, pdfs_hash: str) -> AnalysisResponse:
    """Retrieve cached report from SQLite database."""
    try:
        conn = sqlite3.connect('cache.db')
        cursor = conn.cursor()
        cursor.execute("""
            SELECT report_json, timestamp 
            FROM report_cache 
            WHERE folder_path_hash = ? AND pdfs_hash = ?
        """, (folder_path_hash, pdfs_hash))
        result = cursor.fetchone()
        conn.close()

        if result:
            report_json, timestamp = result
            CACHE_TTL_SECONDS = 3 * 24 * 60 * 60  # 3 days
            if time.time() - timestamp < CACHE_TTL_SECONDS:
                return AnalysisResponse(**json.loads(report_json))
            else:
                logger.info(f"Cache expired for folder_path_hash: {folder_path_hash}")
        return None
    except Exception as e:
        logger.error(f"Error retrieving cached report: {str(e)}")
        return None

def store_cached_report(folder_path_hash: str, pdfs_hash: str, response: AnalysisResponse):
    """Store analysis response in SQLite cache."""
    try:
        with shared_state.lock:
            conn = sqlite3.connect('cache.db')
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO report_cache 
                (folder_path_hash, pdfs_hash, report_json, timestamp)
                VALUES (?, ?, ?, ?)
            """, (folder_path_hash, pdfs_hash, response.json(), time.time()))
            conn.commit()
            conn.close()
    except Exception as e:
        logger.error(f"Error storing cached report: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to store cached report")

async def run_full_analysis(folder_path: str, pdf_files: List[str]) -> AnalysisResponse:
    """Run full analysis pipeline on PDF files."""
    try:
        folder_path = convert_windows_path(folder_path)
        folder_path = os.path.normpath(folder_path)
        versions = ['1.0', '1.1', '2.0']  # Example versions, adjust as needed
        extracted_text = extract_text_from_pdfs(pdf_files)
        data_crew, report_crew, viz_crew = setup_crew(extracted_text, versions)

        logger.info("Starting data crew")
        data_result = await data_crew.kickoff_async()
        logger.info(f"Data crew completed: {data_result}")

        with shared_state.lock:
            if not shared_state.metrics:
                logger.error("No metrics data available after data crew")
                raise HTTPException(status_code=500, detail="Failed to process metrics data")
            metrics_data = shared_state.metrics

        logger.info("Starting report crew")
        report_result = await report_crew.kickoff_async()
        if not isinstance(report_result.raw, str):
            logger.error(f"Report crew returned non-string output: {report_result.raw}")
            raise HTTPException(status_code=500, detail="Report generation failed")
        enhanced_report = enhance_report_markdown(report_result.raw)
        if not validate_report(enhanced_report):
            logger.error(f"Generated report is invalid: {enhanced_report[:200]}...")
            raise HTTPException(status_code=500, detail="Generated report is invalid")

        viz_folder = "visualizations"
        if os.path.exists(viz_folder):
            shutil.rmtree(viz_folder)
        os.makedirs(viz_folder, exist_ok=True)
        run_fallback_visualization(metrics_data)

        viz_base64 = []
        min_visualizations = 5  # Adjust based on EXPECTED_METRICS
        if os.path.exists(viz_folder):
            viz_files = sorted([f for f in os.listdir(viz_folder) if f.endswith('.png')])
            for img in viz_files:
                img_path = os.path.join(viz_folder, img)
                base64_str = get_base64_image(img_path)
                if base64_str:
                    viz_base64.append(base64_str)
            logger.info(f"Generated {len(viz_base64)} visualizations, minimum required {min_visualizations}")
            if len(viz_base64) < min_visualizations:
                logger.error(f"Too few visualizations: {len(viz_base64)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to generate minimum required visualizations: got {len(viz_base64)}, need at least {min_visualizations}"
                )

        evaluation = {
            "score": 85,
            "text": "The analysis indicates stable performance with minor issues in defect closure rates."
        }

        hyperlinks = [
            {
                "url": "http://example.com/report.pdf",
                "page": 1,
                "source_file": os.path.basename(pdf_files[0]) if pdf_files else "unknown.pdf",
                "context": "Reference to defect metrics"
            }
        ]

        response = AnalysisResponse(
            metrics=metrics_data,
            visualizations=viz_base64,
            report=enhanced_report,
            evaluation=evaluation,
            hyperlinks=hyperlinks
        )

        folder_path_hash = hash_string(folder_path)
        pdfs_hash = hash_pdf_contents(pdf_files)
        store_cached_report(folder_path_hash, pdfs_hash, response)

        return response
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        plt.close('all')

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest):
    """Endpoint to analyze PDF reports in a folder."""
    try:
        folder_path = convert_windows_path(request.folder_path)
        folder_path = os.path.normpath(folder_path)
        pdf_files = get_pdf_files_from_folder(folder_path)
        if not pdf_files:
            logger.error(f"No PDF files found in folder: {folder_path}")
            raise HTTPException(status_code=400, detail="No PDF files found in the specified folder")

        folder_path_hash = hash_string(folder_path)
        pdfs_hash = hash_pdf_contents(pdf_files)
        cached_response = get_cached_report(folder_path_hash, pdfs_hash)
        if cached_response:
            logger.info(f"Returning cached report for folder_path_hash: {folder_path_hash}")
            return cached_response

        return await run_full_analysis(folder_path, pdf_files)
    except Exception as e:
        logger.error(f"Error in /analyze endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save-report")
async def save_report(request: SaveReportRequest):
    """Endpoint to save edited report and updated metrics, regenerating visualizations."""
    try:
        folder_path = convert_windows_path(request.folder_path)
        folder_path = os.path.normpath(folder_path)
        folder_path_hash = hash_string(folder_path)
        pdf_files = get_pdf_files_from_folder(folder_path)
        pdfs_hash = hash_pdf_contents(pdf_files)
        
        # Retrieve existing cached response
        cached_response = get_cached_report(folder_path_hash, pdfs_hash)
        if not cached_response:
            raise HTTPException(status_code=404, detail="No cached report found for the provided folder path")
        
        # Update shared_state with new metrics
        with shared_state.lock:
            shared_state.metrics = request.metrics

        # Regenerate visualizations
        viz_folder = "visualizations"
        if os.path.exists(viz_folder):
            shutil.rmtree(viz_folder)
        os.makedirs(viz_folder, exist_ok=True)
        run_fallback_visualization(shared_state.metrics)
        
        # Collect new visualizations
        viz_base64 = []
        min_visualizations = 5  # Adjust based on EXPECTED_METRICS
        if os.path.exists(viz_folder):
            viz_files = sorted([f for f in os.listdir(viz_folder) if f.endswith('.png')])
            for img in viz_files:
                img_path = os.path.join(viz_folder, img)
                base64_str = get_base64_image(img_path)
                if base64_str:
                    viz_base64.append(base64_str)
            logger.info(f"Generated {len(viz_base64)} visualizations, minimum required {min_visualizations}")
            if len(viz_base64) < min_visualizations:
                logger.error(f"Too few visualizations: {len(viz_base64)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to generate minimum required visualizations: got {len(viz_base64)}, need at least {min_visualizations}"
                )

        # Update cached response
        updated_response = AnalysisResponse(
            metrics=shared_state.metrics,
            visualizations=viz_base64,
            report=request.report,  # Store the fully edited report
            evaluation=cached_response.evaluation,  # Preserve original evaluation
            hyperlinks=cached_response.hyperlinks  # Preserve original hyperlinks
        )
        
        # Store updated response in cache
        store_cached_report(folder_path_hash, pdfs_hash, updated_response)
        logger.info(f"Updated cached report for folder_path_hash: {folder_path_hash}")
        
        return {"status": "success", "message": "Report and visualizations updated in cache"}
    except Exception as e:
        logger.error(f"Error in /save-report endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        plt.close('all')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
