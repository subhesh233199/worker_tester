import os
import re
import json
import runpy
import base64
import hashlib
import sqlite3
import time
from typing import List, Dict, Tuple, Any, Union
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from crewai import Agent, Task, Crew, Process, LLM
from langchain_openai import AzureChatOpenAI
import ssl
import warnings
import shutil
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, validator
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tenacity import retry, stop_after_attempt, wait_fixed
from copy import deepcopy

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

# Constants from TM_crew_setup.py
EXPECTED_METRICS = [
    "Open ALL RRR Defects", "Open Security Defects", "All Open Defects (T-1)",
    "All Security Open Defects", "Load/Performance", "E2E Test Coverage",
    "Automation Test Coverage", "Unit Test Coverage", "Defect Closure Rate",
    "Regression Issues", "Customer Specific Testing (UAT)"
]
VALID_STATUSES = ['ON TRACK', 'MEDIUM RISK', 'RISK', 'NEEDS REVIEW']

def parse_markdown_table(table_text: str, is_uat: bool = False) -> List[Dict[str, Any]]:
    """Parse a markdown table into a list of dictionaries."""
    try:
        lines = [line.strip() for line in table_text.split('\n') if line.strip()]
        if len(lines) < 2:
            logger.warning("Table has insufficient lines")
            return []

        # Extract headers
        headers = [h.strip() for h in lines[0].split('|') if h.strip()]
        if not headers:
            logger.warning("No headers found in table")
            return []

        # Skip separator line (e.g., |---|---|)
        if not re.match(r'^\|[-:| ]+\|$', lines[1]):
            logger.warning("Invalid table separator line")
            return []

        # Expected headers
        if is_uat:
            expected_headers = ['Release', 'Pass Count', 'Fail Count', 'Pass Rate (%)', 'Trend', 'Status']
            if not all(h in headers for h in ['Release', 'Pass Count', 'Fail Count', 'Status']):
                logger.warning(f"Missing required UAT headers: {headers}")
                return []
        else:
            expected_headers = ['Release', 'Value', 'Trend', 'Status']
            if not all(h in headers for h in ['Release', 'Value', 'Status']):
                logger.warning(f"Missing required headers: {headers}")
                return []

        # Parse rows
        result = []
        for line in lines[2:]:
            values = [v.strip() for v in line.split('|') if v.strip()]
            if len(values) != len(headers):
                logger.warning(f"Mismatched columns in row: {line}")
                continue

            row = {}
            for header, value in zip(headers, values):
                if header == 'Release':
                    if not re.match(r'^\d+\.\d+$', value):
                        logger.warning(f"Invalid version format: {value}")
                        continue
                    row['version'] = value
                elif header in ['Pass Count', 'Fail Count', 'Value']:
                    try:
                        num = float(value)
                        if num < 0:
                            logger.warning(f"Negative value: {value}")
                            continue
                        row[header.lower().replace(' ', '_')] = int(num) if num.is_integer() else num
                    except ValueError:
                        logger.warning(f"Invalid number: {value}")
                        continue
                elif header == 'Status':
                    # Remove bold markdown (e.g., **ON TRACK** -> ON TRACK)
                    clean_value = re.sub(r'\*\*(.*?)\*\*', r'\1', value)
                    if clean_value not in VALID_STATUSES:
                        logger.warning(f"Invalid status: {clean_value}")
                        continue
                    row['status'] = clean_value
                elif header == 'Trend' and value:
                    if not re.match(r'^(↑|↓)\s*\(\d+\.\d+%\)|→$', value):
                        logger.warning(f"Invalid trend: {value}")
                    else:
                        row['trend'] = value

            if 'version' in row and 'status' in row and (is_uat.get('is_uat', False) and 'pass_count' in row and 'fail_count' in row or not is_uat and 'value' in row):
                result.append(row)
            else:
                logger.warning(f"Incomplete row: {row}")
        return result
    except Exception as e:
        logger.error(f"Error parsing table: {str(e)}")
        return []

def extract_metrics_from_markdown(md_text: str) -> Dict[str, Any]:
    """Extract metrics data from the Metrics Summary section of markdown."""
    metrics = {'metrics': {}}
    
    # Split into sections
    sections = re.split(r'^##\s(.+?)\n', md_text, flags=re.MULTILINE)
    metrics_summary = ''
    for i in range(1, len(sections), 2):
        if sections[i].strip() == 'Metrics Summary':
            metrics_summary = sections[i + 1]
            break

    if not metrics_summary:
        logger.warning("Metrics Summary section not found")
        return metrics

    # Define submetrics for ATLS/BTLS
    atls_btls_metrics = EXPECTED_METRICS[:5]
    uat_metric = 'Customer Specific Testing (UAT)'
    other_metrics = [m for m in EXPECTEDMETRICS if m not in atls_btls_metrics and m != uat_metric]

    # Split Metrics Summary into subsections
    subsections = re.split(r'^###\s+(.+)\n', metrics_summary, flags=re.MULTILINE)
    current_metric = None
    current_submetric = None

    for i in range(1, subsections), len(subsections, 2)):
        header = subsections[i].strip()
        content = subsections[i+1]
        
        # Handle UAT submetrics (RBS, Tesco, Belk)
        if current_metric == 'Customer Specific Testing (UAT)' and header in ['RBS', 'Tesco', 'Belk']:
            current_submetric = header
            table_data = parse_markdown_table(content, is_uat=True)
            if table_data:
                if uat_metric not in metrics['metrics']:
                    metrics['metrics'][uat_metric] = {}
                metrics['metrics'][uat_metric][current_submetric] = table_data
                logger.info(f"Parsed {len(table_data)} rows for {uat_metric} {current_submetric}")
            continue
        
        # Handle ATLS/BTLS submetrics
        match = re.match(r'(.+)\s+\((ATLS|BTLS)\)$', header)
        if match:
            metric_name, submetric = match.groups()
            if metric_name in atls_btls_metrics:
                current_metric = metric_name
                current_submetric = submetric
                table_data = parse_markdown_table(content)
                if table_data:
                    if current_metric not in metrics['metrics']:
                        metrics['metrics'][current_metric] = {}
                    metrics['metrics'][current_metric][current_submetric] = table_data
                    logger.info(f"Parsed {len(table_data)} rows for {current_metric} {current_submetric}")
                continue
        
        # Handle other metrics
        if header in other_metrics:
            current_metric = header
            current_submetric = None
            table_data = parse_markdown_table(content)
            if table_data:
                metrics['metrics'][current_metric] = table_data
                logger.info(f"Parsed {len(table_data)} rows for {current_metric}")
            continue
        
        # Set current_metric for UAT
        if header == 'Customer Specific Testing (UAT)':
            current_metric = header
            current_submetric = None

    return metrics

def enhance_report_markdown(md_text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Enhance the markdown report and extract metrics from the Metrics Summary section.

    Args:
        md_text (str): Raw markdown text

    Returns:
        Tuple[str, Dict[str, Any]]: Enhanced markdown text and parsed metrics
    """
    try:
        # Remove markdown code fences
        cleaned = re.sub(r'^```markdown\n|\n```$', '', md_text, flags=re.MULTILINE)
        
        # Fix table alignment (remove extra spaces, ensure proper pipes)
        cleaned = re.sub(r'(\|.+\|)\n\s*(\|-+\|)', r'\1\n\2', cleaned)
        
        # Clean invalid trend symbols (e.g., '4', 't', '/')
        cleaned = re.sub(r'\b[4t/]\b', '→', cleaned)  # Replace stray symbols with arrow
        cleaned = re.sub(r'\s*\|\s*', ' | ', cleaned)  # Normalize spacing around pipes
        cleaned = re.sub(r'\s{2,}', ' ', cleaned)  # Collapse multiple spaces
        
        # Enhance statuses
        status_map = {
            "MEDIUM RISK": "**MEDIUM RISK**",
            "HIGH RISK": "**HIGH RISK**",
            "LOW RISK": "**LOW RISK**",
            "ON TRACK": "**ON TRACK**",
            "RISK": "**RISK**",
            "NEEDS REVIEW": "**NEEDS REVIEW**"
        }
        for k, v in status_map.items():
            cleaned = cleaned.replace(k, v)
        
        # Fix headers and list items
        cleaned = re.sub(r'^#\s+(.+)$', r'# \1\n', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'^##\s+(.+)$', r'## \1\n', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'^\s*-\s+(.+)', r'- \1', cleaned, flags=re.MULTILINE)
        
        # Extract metrics
        metrics = extract_metrics_from_markdown(cleaned)
        
        return cleaned.encode('utf-8').decode('utf-8'), metrics
    
    except Exception as e:
        logger.error(f"Error enhancing markdown: {str(e)}")
        return md_text.encode('utf-8').decode('utf-8'), {'metrics': {}}
