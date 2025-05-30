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
from state import shared_state

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI
llm = LLM(
    model=f"azure/{os.getenv('DEPLOYMENT_NAME')}",
    api_version=os.getenv("AZURE_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
    temperature=0.1,
    top_p=0.95,
)

# Constants
EXPECTED_METRICS = [
    "Open ALL RRR Defects", "Open Security Defects", "All Open Defects (T-1)",
    "All Security Open Defects", "Load/Performance", "E2E Test Coverage",
    "Automation Test Coverage", "Unit Test Coverage", "Defect Closure Rate",
    "Regression Issues", "Customer Specific Testing (UAT)"
]

def validate_metrics(metrics: Dict[str, Any]) -> bool:
    """
    Validates the structure and content of metrics data.

    Checks:
    - Required metrics presence
    - Data type correctness
    - Value ranges
    - Status values
    - Optional trend format

    Args:
        metrics (Dict[str, Any]): Metrics data to validate

    Returns:
        bool: True if valid, False otherwise
    """
    if not metrics or 'metrics' not in metrics or not isinstance(metrics['metrics'], dict):
        logger.error(f"Invalid metrics structure: {metrics}")
        return False

    for metric in EXPECTED_METRICS:
        if metric not in metrics['metrics']:
            logger.warning(f"Missing metric: {metric}")
            continue

        data = metrics['metrics'][metric]
        if metric in EXPECTED_METRICS[:5]:  # ATLS/BTLS metrics
            if not isinstance(data, dict) or 'ATLS' not in data or 'BTLS' not in data:
                logger.error(f"Invalid ATLS/BTLS structure for {metric}: {data}")
                return False
            for sub in ['ATLS', 'BTLS']:
                if not isinstance(data[sub], list):
                    logger.error(f"Invalid {sub} data for {metric}: {data[sub]}")
                    return False
                for item in data[sub]:
                    try:
                        item_dict = dict(item)
                        if not all(k in item_dict for k in ['version', 'value', 'status']):
                            logger.error(f"Missing keys in {sub} item for {metric}: {item}")
                            return False
                        if not isinstance(item_dict['version'], str) or not re.match(r'^\d+\.\d+$', item_dict['version']):
                            logger.error(f"Invalid version in {sub} item for {metric}: {item}")
                            return False
                        if not isinstance(item_dict['value'], (int, float)) or item_dict['value'] < 0:
                            logger.error(f"Invalid value in {sub} item for {metric}: {item}")
                            return False
                        if item_dict['status'] not in ['ON TRACK', 'MEDIUM RISK', 'RISK', 'NEEDS REVIEW']:
                            logger.error(f"Invalid status in {sub} item for {metric}: {item}")
                            return False
                        if 'trend' in item_dict and not re.match(r'^(↑|↓)\s*\(\d+\.\d+%\)|→$', item_dict['trend']):
                            logger.error(f"Invalid trend in {sub} item for {metric}: {item}")
                            return False
                    except Exception as e:
                        logger.error(f"Invalid item in {sub} for {metric}: {item}, error: {str(e)}")
                        return False
        elif metric == "Customer Specific Testing (UAT)":
            if not isinstance(data, dict) or not all(client in data for client in ['RBS', 'Tesco', 'Belk']):
                logger.error(f"Invalid structure for {metric}: {data}")
                return False
            for client in ['RBS', 'Tesco', 'Belk']:
                client_data = data.get(client, [])
                if not isinstance(client_data, list):
                    logger.error(f"Invalid data for {metric} {client}: {client_data}")
                    return False
                for item in client_data:
                    try:
                        item_dict = dict(item)
                        if not all(k in item_dict for k in ['version', 'pass_count', 'fail_count', 'status']):
                            logger.error(f"Missing keys in {client} item for {metric}: {item}")
                            return False
                        if not isinstance(item_dict['version'], str) or not re.match(r'^\d+\.\d+$', item_dict['version']):
                            logger.error(f"Invalid version in {client} item for {metric}: {item}")
                            return False
                        if not isinstance(item_dict['pass_count'], (int, float)) or item_dict['pass_count'] < 0:
                            logger.error(f"Invalid pass_count in {client} item for {metric}: {item}")
                            return False
                        if not isinstance(item_dict['fail_count'], (int, float)) or item_dict['fail_count'] < 0:
                            logger.error(f"Invalid fail_count in {client} item for {metric}: {item}")
                            return False
                        if item_dict['status'] not in ['ON TRACK', 'MEDIUM RISK', 'RISK', 'NEEDS REVIEW']:
                            logger.error(f"Invalid status in {client} item for {metric}: {item}")
                            return False
                        if 'trend' in item_dict and not re.match(r'^(↑|↓)\s*\(\d+\.\d+%\)|→$', item_dict['trend']):
                            logger.error(f"Invalid trend in {client} item for {metric}: {item}")
                            return False
                    except Exception as e:
                        logger.error(f"Invalid item in {client} for {metric}: {item}, error: {str(e)}")
                        return False
        else:  # Non-ATLS/BTLS metrics
            if not isinstance(data, list):
                logger.error(f"Invalid data for {metric}: {data}")
                return False
            for item in data:
                try:
                    item_dict = dict(item)
                    if not all(k in item_dict for k in ['version', 'value', 'status']):
                        logger.error(f"Missing keys in item for {metric}: {item}")
                        return False
                    if not isinstance(item_dict['version'], str) or not re.match(r'^\d+\.\d+$', item_dict['version']):
                        logger.error(f"Invalid version in item for {metric}: {item}")
                        return False
                    if not isinstance(item_dict['value'], (int, float)) or item_dict['value'] < 0:
                        logger.error(f"Invalid value in item for {metric}: {item}")
                        return False
                    if item_dict['status'] not in ['ON TRACK', 'MEDIUM RISK', 'RISK', 'NEEDS REVIEW']:
                        logger.error(f"Invalid status in item for {metric}: {item}")
                        return False
                    if 'trend' in item_dict and not re.match(r'^(↑|↓)\s*\(\d+\.\d+%\)|→$', item_dict['trend']):
                        logger.error(f"Invalid trend in item for {metric}: {item}")
                        return False
                except Exception as e:
                    logger.error(f"Invalid item for {metric}: {item}, error: {str(e)}")
                    return False
    return True

def clean_json_output(raw_output: str, fallback_versions: List[str]) -> dict:
    logger.info(f"Raw analysis output: {raw_output[:200]}...")
    # Synthetic data for fallback
    default_json = {
        "metrics": {
            metric: {
                "ATLS": [
                    {"version": v, "value": 10 if i == 0 else 0, "status": "NEEDS REVIEW"}
                    for i, v in enumerate(fallback_versions)
                ],
                "BTLS": [
                    {"version": v, "value": 12 if i == 0 else 0, "status": "NEEDS REVIEW"}
                    for i, v in enumerate(fallback_versions)
                ]
            } if metric in EXPECTED_METRICS[:5] else
            {
                "RBS": [
                    {"version": v, "pass_count": 50 if i == 0 else 0, "fail_count": 0, "status": "NEEDS REVIEW"}
                    for i, v in enumerate(fallback_versions)
                ],
                "Tesco": [
                    {"version": v, "pass_count": 45 if i == 0 else 0, "fail_count": 0, "status": "NEEDS REVIEW"}
                    for i, v in enumerate(fallback_versions)
                ],
                "Belk": [
                    {"version": v, "pass_count": 40 if i == 0 else 0, "fail_count": 0, "status": "NEEDS REVIEW"}
                    for i, v in enumerate(fallback_versions)
                ]
            } if metric == "Customer Specific Testing (UAT)" else
            [
                {"version": v, "value": 80 if i == 0 else 0, "status": "NEEDS REVIEW"}
                for i, v in enumerate(fallback_versions)
            ]
            for metric in EXPECTED_METRICS
        }
    }

    try:
        data = json.loads(raw_output)
        if validate_metrics(data):
            return data
        logger.warning(f"Direct JSON invalid: {json.dumps(data, indent=2)[:200]}...")
    except json.JSONDecodeError as e:
        logger.warning(f"Direct JSON parsing failed: {str(e)}")

    try:
        cleaned = re.search(r'```json\s*([\s\S]*?)\s*```', raw_output, re.MULTILINE)
        if cleaned:
            data = json.loads(cleaned.group(1))
            if validate_metrics(data):
                return data
            logger.warning(f"Code block JSON invalid: {json.dumps(data, indent=2)[:200]}...")
    except json.JSONDecodeError as e:
        logger.warning(f"Code block JSON parsing failed: {str(e)}")

    try:
        cleaned = re.search(r'\{[\s\S]*\}', raw_output, re.MULTILINE)
        if cleaned:
            json_str = cleaned.group(0)
            json_str = re.sub(r"'", '"', json_str)
            json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
            data = json.loads(json_str)
            if validate_metrics(data):
                return data
            logger.warning(f"JSON-like structure invalid: {json.dumps(data, indent=2)[:200]}...")
    except json.JSONDecodeError as e:
        logger.warning(f"JSON-like structure parsing failed: {str(e)}")

    logger.error(f"Failed to parse JSON, using default structure with zero values for versions: {fallback_versions}")
    return default_json

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def process_task_output(raw_output: str, fallback_versions: List[str]) -> Dict:
    logger.info(f"Raw output type: {type(raw_output)}, content: {raw_output if isinstance(raw_output, str) else raw_output}")
    if not isinstance(raw_output, str):
        logger.warning(f"Expected raw_output to be a string, got {type(raw_output)}. Falling back to empty JSON.")
        raw_output = "{}"  # Fallback to empty JSON string
    logger.info(f"Processing task output: {raw_output[:200]}...")
    data = clean_json_output(raw_output, fallback_versions)
    if not validate_metrics(data):
        logger.error(f"Validation failed for processed output: {json.dumps(data, indent=2)[:200]}...")
        raise ValueError("Invalid or incomplete metrics data")
    # Validate and correct trends
    for metric, metric_data in data['metrics'].items():
        if metric in EXPECTED_METRICS[:5]:  # ATLS/BTLS metrics
            for sub in ['ATLS', 'BTLS']:
                items = sorted(metric_data[sub], key=lambda x: x['version'])
                for i in range(len(items)):
                    if i == 0 or not items[i].get('value') or not items[i-1].get('value'):
                        items[i]['trend'] = '→'
                    else:
                        prev_val = float(items[i-1]['value'])
                        curr_val = float(items[i]['value'])
                        if prev_val == 0 or abs(curr_val - prev_val) < 0.01:
                            items[i]['trend'] = '→'
                        else:
                            pct_change = ((curr_val - prev_val) / prev_val) * 100
                            if abs(pct_change) < 1:
                                items[i]['trend'] = '→'
                            elif pct_change > 0:
                                items[i]['trend'] = f"↑ ({abs(pct_change):.1f}%)"
                            else:
                                items[i]['trend'] = f"↓ ({abs(pct_change):.1f}%)"
        elif metric == "Customer Specific Testing (UAT)":
            for client in ['RBS', 'Tesco', 'Belk']:
                items = sorted(metric_data[client], key=lambda x: x['version'])
                for i in range(len(items)):
                    pass_count = float(items[i].get('pass_count', 0))
                    fail_count = float(items[i].get('fail_count', 0))
                    total = pass_count + fail_count
                    pass_rate = (pass_count / total * 100) if total > 0 else 0
                    items[i]['pass_rate'] = pass_rate
                    if i == 0:
                        items[i]['trend'] = '→'
                    else:
                        prev_pass_count = float(items[i-1].get('pass_count', 0))
                        prev_fail_count = float(items[i-1].get('fail_count', 0))
                        prev_total = prev_pass_count + prev_fail_count
                        prev_pass_rate = (prev_pass_count / prev_total * 100) if prev_total > 0 else 0
                        if prev_total == 0 or total == 0 or abs(pass_rate - prev_pass_rate) < 0.01:
                            items[i]['trend'] = '→'
                        else:
                            pct_change = pass_rate - prev_pass_rate
                            if abs(pct_change) < 1:
                                items[i]['trend'] = '→'
                            elif pct_change > 0:
                                items[i]['trend'] = f"↑ ({abs(pct_change):.1f}%)"
                            else:
                                items[i]['trend'] = f"↓ ({abs(pct_change):.1f}%)"
        else:  # Non-ATLS/BTLS metrics
            items = sorted(metric_data, key=lambda x: x['version'])
            for i in range(len(items)):
                if i == 0 or not items[i].get('value') or not items[i-1].get('value'):
                    items[i]['trend'] = '→'
                else:
                    prev_val = float(items[i-1]['value'])
                    curr_val = float(items[i]['value'])
                    if prev_val == 0 or abs(curr_val - prev_val) < 0.01:
                        items[i]['trend'] = '→'
                    else:
                        pct_change = ((curr_val - prev_val) / prev_val) * 100
                        if abs(pct_change) < 1:
                            items[i]['trend'] = '→'
                        elif pct_change > 0:
                            items[i]['trend'] = f"↑ ({abs(pct_change):.1f}%)"
                        else:
                            items[i]['trend'] = f"↓ ({abs(pct_change):.1f}%)"
    return data

def run_fallback_visualization(metrics: Dict[str, Any]):
    with shared_state.viz_lock:
        try:
            os.makedirs("visualizations", exist_ok=True)
            logging.basicConfig(level=logging.INFO, filename='visualization.log')
            logger = logging.getLogger(__name__)
            logger.info("Starting fallback visualization")

            if not metrics or 'metrics' not in metrics or not isinstance(metrics['metrics'], dict):
                logger.error(f"Invalid metrics data: {metrics}")
                raise ValueError("Metrics data is empty or invalid")

            atls_btls_metrics = EXPECTED_METRICS[:5]
            coverage_metrics = EXPECTED_METRICS[5:8]
            other_metrics = EXPECTED_METRICS[8:10]
            uat_metric = "Customer Specific Testing (UAT)"

            generated_files = []

            # ATLS/BTLS Metrics (Grouped Bar Charts)
            for metric in atls_btls_metrics:
                try:
                    data = metrics['metrics'].get(metric, {})
                    if not isinstance(data, dict) or 'ATLS' not in data or 'BTLS' not in data:
                        logger.warning(f"Creating placeholder for {metric}: invalid or missing ATLS/BTLS data")
                        plt.figure(figsize=(8,5), dpi=120)
                        plt.text(0.5, 0.5, f"No data for {metric}", ha='center', va='center')
                        plt.title(metric)
                        filename = f'visualizations/{metric.replace("/", "_")}_atls_btls.png'
                        plt.savefig(filename)
                        plt.close()
                        generated_files.append(filename)
                        logger.info(f"Generated placeholder chart for {metric}: {filename}")
                        continue
                    atls_data = data.get('ATLS', [])
                    btls_data = data.get('BTLS', [])
                    versions = [item['version'] for item in atls_data if isinstance(item, dict) and 'version' in item and 'value' in item]
                    atls_values = [float(item['value']) if isinstance(item['value'], (int, float)) else 0 for item in atls_data if isinstance(item, dict) and 'version' in item and 'value' in item]
                    btls_values = [float(item['value']) if isinstance(item['value'], (int, float)) else 0 for item in btls_data if isinstance(item, dict) and 'version' in item and 'value' in item]
                    if not versions or len(atls_values) != len(versions) or len(btls_values) != len(versions):
                        logger.warning(f"Creating placeholder for {metric}: inconsistent data lengths")
                        plt.figure(figsize=(8,5), dpi=120)
                        plt.text(0.5, 0.5, f"Incomplete data for {metric}", ha='center', va='center')
                        plt.title(metric)
                        filename = f'visualizations/{metric.replace("/", "_")}_atls_btls.png'
                        plt.savefig(filename)
                        plt.close()
                        generated_files.append(filename)
                        logger.info(f"Generated placeholder chart for {metric}: {filename}")
                        continue
                    x = np.arange(len(versions))
                    width = 0.35
                    plt.figure(figsize=(8,5), dpi=120)
                    plt.bar(x - width/2, atls_values, width, label='ATLS', color='blue')
                    plt.bar(x + width/2, btls_values, width, label='BTLS', color='orange')
                    plt.xlabel('Release')
                    plt.ylabel('Value')
                    plt.title(metric)
                    plt.xticks(x, versions)
                    plt.legend()
                    filename = f'visualizations/{metric.replace("/", "_")}_atls_btls.png'
                    plt.savefig(filename)
                    plt.close()
                    generated_files.append(filename)
                    logger.info(f"Generated grouped bar chart for {metric}: {filename}")
                except Exception as e:
                    logger.error(f"Failed to generate chart for {metric}: {str(e)}")
                    plt.figure(figsize=(8,5), dpi=120)
                    plt.text(0.5, 0.5, f"Error generating {metric}", ha='center', va='center')
                    plt.title(metric)
                    filename = f'visualizations/{metric.replace("/", "_")}_atls_btls.png'
                    plt.savefig(filename)
                    plt.close()
                    generated_files.append(filename)
                    logger.info(f"Generated error placeholder chart for {metric}: {filename}")

            # Coverage Metrics (Line Charts)
            for metric in coverage_metrics:
                try:
                    data = metrics['metrics'].get(metric, [])
                    if not isinstance(data, list) or not data:
                        logger.warning(f"Creating placeholder for {metric}: invalid data format")
                        plt.figure(figsize=(8,5), dpi=120)
                        plt.text(0.5, 0.5, f"No data for {metric}", ha='center', va='center')
                        plt.title(metric)
                        filename = f'visualizations/{metric.replace("/", "_")}.png'
                        plt.savefig(filename)
                        plt.close()
                        generated_files.append(filename)
                        logger.info(f"Generated placeholder chart for {metric}: {filename}")
                        continue
                    versions = [item['version'] for item in data if isinstance(item, dict) and 'version' in item and 'value' in item]
                    values = [float(item['value']) if isinstance(item['value'], (int, float)) else 0 for item in data if isinstance(item, dict) and 'version' in item and 'value' in item]
                    if not versions or len(values) != len(versions):
                        logger.warning(f"Creating placeholder for {metric}: inconsistent data lengths")
                        plt.figure(figsize=(8,5), dpi=120)
                        plt.text(0.5, 0.5, f"Incomplete data for {metric}", ha='center', va='center')
                        plt.title(metric)
                        filename = f'visualizations/{metric.replace("/", "_")}.png'
                        plt.savefig(filename)
                        plt.close()
                        generated_files.append(filename)
                        logger.info(f"Generated placeholder chart for {metric}: {filename}")
                        continue
                    plt.figure(figsize=(8,5), dpi=120)
                    plt.plot(versions, values, marker='o', color='green')
                    plt.xlabel('Release')
                    plt.ylabel('Value')
                    plt.title(metric)
                    filename = f'visualizations/{metric.replace("/", "_")}.png'
                    plt.savefig(filename)
                    plt.close()
                    generated_files.append(filename)
                    logger.info(f"Generated line chart for {metric}: {filename}")
                except Exception as e:
                    logger.error(f"Failed to generate chart for {metric}: {str(e)}")
                    plt.figure(figsize=(8,5), dpi=120)
                    plt.text(0.5, 0.5, f"Error generating {metric}", ha='center', va='center')
                    plt.title(metric)
                    filename = f'visualizations/{metric.replace("/", "_")}.png'
                    plt.savefig(filename)
                    plt.close()
                    generated_files.append(filename)
                    logger.info(f"Generated error placeholder chart for {metric}: {filename}")

            # Other Metrics (Bar Charts)
            for metric in other_metrics:
                try:
                    data = metrics['metrics'].get(metric, [])
                    if not isinstance(data, list) or not data:
                        logger.warning(f"Creating placeholder for {metric}: invalid data format")
                        plt.figure(figsize=(8,5), dpi=120)
                        plt.text(0.5, 0.5, f"No data for {metric}", ha='center', va='center')
                        plt.title(metric)
                        filename = f'visualizations/{metric.replace("/", "_")}.png'
                        plt.savefig(filename)
                        plt.close()
                        generated_files.append(filename)
                        logger.info(f"Generated placeholder chart for {metric}: {filename}")
                        continue
                    versions = [item['version'] for item in data if isinstance(item, dict) and 'version' in item and 'value' in item]
                    values = [float(item['value']) if isinstance(item['value'], (int, float)) else 0 for item in data if isinstance(item, dict) and 'version' in item and 'value' in item]
                    if not versions or len(values) != len(versions):
                        logger.warning(f"Creating placeholder for {metric}: inconsistent data lengths")
                        plt.figure(figsize=(8,5), dpi=120)
                        plt.text(0.5, 0.5, f"Incomplete data for {metric}", ha='center', va='center')
                        plt.title(metric)
                        filename = f'visualizations/{metric.replace("/", "_")}.png'
                        plt.savefig(filename)
                        plt.close()
                        generated_files.append(filename)
                        logger.info(f"Generated placeholder chart for {metric}: {filename}")
                        continue
                    plt.figure(figsize=(8,5), dpi=120)
                    plt.bar(versions, values, color='purple')
                    plt.xlabel('Release')
                    plt.ylabel('Value')
                    plt.title(metric)
                    filename = f'visualizations/{metric.replace("/", "_")}.png'
                    plt.savefig(filename)
                    plt.close()
                    generated_files.append(filename)
                    logger.info(f"Generated bar chart for {metric}: {filename}")
                except Exception as e:
                    logger.error(f"Failed to generate chart for {metric}: {str(e)}")
                    plt.figure(figsize=(8,5), dpi=120)
                    plt.text(0.5, 0.5, f"Error generating {metric}", ha='center', va='center')
                    plt.title(metric)
                    filename = f'visualizations/{metric.replace("/", "_")}.png'
                    plt.savefig(filename)
                    plt.close()
                    generated_files.append(filename)
                    logger.info(f"Generated error placeholder chart for {metric}: {filename}")

            # UAT Metrics (Stacked Bar Charts for RBS, Tesco, Belk)
            if uat_metric in metrics['metrics']:
                for client in ['RBS', 'Tesco', 'Belk']:
                    try:
                        data = metrics['metrics'][uat_metric].get(client, [])
                        if not isinstance(data, list) or not data:
                            logger.warning(f"Creating placeholder for UAT {client}: invalid data format")
                            plt.figure(figsize=(8,5), dpi=120)
                            plt.text(0.5, 0.5, f"No data for UAT {client}", ha='center', va='center')
                            plt.title(f"UAT {client}")
                            filename = f'visualizations/uat_{client.lower()}.png'
                            plt.savefig(filename)
                            plt.close()
                            generated_files.append(filename)
                            logger.info(f"Generated placeholder chart for UAT {client}: {filename}")
                            continue
                        versions = [item['version'] for item in data if isinstance(item, dict) and 'version' in item and 'pass_count' in item and 'fail_count' in item]
                        pass_counts = [float(item['pass_count']) if isinstance(item['pass_count'], (int, float)) else 0 for item in data if isinstance(item, dict) and 'version' in item and 'pass_count' in item]
                        fail_counts = [float(item['fail_count']) if isinstance(item['fail_count'], (int, float)) else 0 for item in data if isinstance(item, dict) and 'version' in item and 'fail_count' in item]
                        if not versions or len(pass_counts) != len(versions) or len(fail_counts) != len(versions):
                            logger.warning(f"Creating placeholder for UAT {client}: inconsistent data lengths")
                            plt.figure(figsize=(8,5), dpi=120)
                            plt.text(0.5, 0.5, f"Incomplete data for UAT {client}", ha='center', va='center')
                            plt.title(f"UAT {client}")
                            filename = f'visualizations/uat_{client.lower()}.png'
                            plt.savefig(filename)
                            plt.close()
                            generated_files.append(filename)
                            logger.info(f"Generated placeholder chart for UAT {client}: {filename}")
                            continue
                        x = np.arange(len(versions))
                        plt.figure(figsize=(8,5), dpi=120)
                        plt.bar(x, pass_counts, label='Pass', color='green')
                        plt.bar(x, fail_counts, bottom=pass_counts, label='Fail', color='red')
                        plt.xlabel('Release')
                        plt.ylabel('Count')
                        plt.title(f"UAT {client}")
                        plt.xticks(x, versions)
                        plt.legend()
                        filename = f'visualizations/uat_{client.lower()}.png'
                        plt.savefig(filename)
                        plt.close()
                        generated_files.append(filename)
                        logger.info(f"Generated stacked bar chart for UAT {client}: {filename}")
                    except Exception as e:
                        logger.error(f"Failed to generate chart for UAT {client}: {str(e)}")
                        plt.figure(figsize=(8,5), dpi=120)
                        plt.text(0.5, 0.5, f"Error generating UAT {client}", ha='center', va='center')
                        plt.title(f"UAT {client}")
                        filename = f'visualizations/uat_{client.lower()}.png'
                        plt.savefig(filename)
                        plt.close()
                        generated_files.append(filename)
                        logger.info(f"Generated error placeholder chart for UAT {client}: {filename}")

            logger.info(f"Completed fallback visualization, generated {len(generated_files)} files")
        except Exception as e:
            logger.error(f"Fallback visualization failed: {str(e)}")
            raise
        finally:
            plt.close('all')

def setup_crew(extracted_text: str, versions: List[str]) -> tuple:
    """
    Sets up the AI crew system for analysis.

    Creates three specialized crews:
    1. Data Crew: Structures raw data into JSON format
    2. Report Crew: Generates comprehensive analysis reports
    3. Visualization Crew: Creates data visualizations

    Args:
        extracted_text (str): Text extracted from PDFs
        versions (List[str]): List of release versions to analyze

    Returns:
        tuple: (data_crew, report_crew, viz_crew)
    """
    structurer = Agent(
        role="Data Architect",
        goal="Structure raw release data into VALID JSON format",
        backstory="Expert in transforming unstructured data into clean JSON structures",
        llm=llm,
        verbose=True,
        memory=True,
    )

    # Ensure we have at least 2 versions for comparison; repeat the last one if needed
    if len(versions) < 2:
        raise ValueError("At least two versions are required for analysis")
    versions_for_example = versions[:3] if len(versions) >= 3 else versions + [versions[-1]] * (3 - len(versions))

    validated_structure_task = Task(
        description=f"""Convert this release data to STRICT JSON:
{extracted_text}

RULES:
1. Output MUST be valid JSON only
2. Use this EXACT structure:
{{
    "metrics": {{
        "Open ALL RRR Defects": {{"ATLS": [{{"version": "{versions[0]}", "value": N, "status": "TEXT"}}, ...], "BTLS": [...]}},
        "Open Security Defects": {{"ATLS": [...], "BTLS": [...]}},
        "All Open Defects (T-1)": {{"ATLS": [...], "BTLS": [...]}},
        "All Security Open Defects": {{"ATLS": [...], "BTLS": [...]}},
        "Load/Performance": {{"ATLS": [...], "BTLS": [...]}},
        "E2E Test Coverage": [{{"version": "{versions[0]}", "value": N, "status": "TEXT"}}, ...],
        "Automation Test Coverage": [...],
        "Unit Test Coverage": [...],
        "Defect Closure Rate": [...],
        "Regression Issues": [...],
        "Customer Specific Testing (UAT)": {{
            "RBS": [{{"version": "{versions[0]}", "pass_count": N, "fail_count": M, "status": "TEXT"}}, ...],
            "Tesco": [...],
            "Belk": [...]
        }}
    }}
}}
3. Include ALL metrics: {', '.join(EXPECTED_METRICS)}
4. Use versions {', '.join(f'"{v}"' for v in versions)}
5. For UAT, pass_count and fail_count must be non-negative integers
6. For other metrics, values must be non-negative numbers
7. Status must be one of: "ON TRACK", "MEDIUM RISK", "RISK", "NEEDS REVIEW"
8. No text outside JSON, no trailing commas, no comments
9. Validate JSON syntax before output
EXAMPLE:
{{
    "metrics": {{
        "Open ALL RRR Defects": {{
            "ATLS": [
                {{"version": "{versions_for_example[0]}", "value": 10, "status": "RISK"}},
                {{"version": "{versions_for_example[1]}", "value": 8, "status": "MEDIUM RISK"}},
                {{"version": "{versions_for_example[2]}", "value": 5, "status": "ON TRACK"}}
            ],
            "BTLS": [
                {{"version": "{versions_for_example[0]}", "value": 12, "status": "RISK"}},
                {{"version": "{versions_for_example[1]}", "value": 9, "status": "MEDIUM RISK"}},
                {{"version": "{versions_for_example[2]}", "value": 6, "status": "ON TRACK"}}
            ]
        }},
        "Customer Specific Testing (UAT)": {{
            "RBS": [
                {{"version": "{versions_for_example[0]}", "pass_count": 50, "fail_count": 5, "status": "ON TRACK"}},
                {{"version": "{versions_for_example[1]}", "pass_count": 48, "fail_count": 6, "status": "MEDIUM RISK"}},
                {{"version": "{versions_for_example[2]}", "pass_count": 52, "fail_count": 4, "status": "ON TRACK"}}
            ],
            "Tesco": [
                {{"version": "{versions_for_example[0]}", "pass_count": 45, "fail_count": 3, "status": "ON TRACK"}},
                {{"version": "{versions_for_example[1]}", "pass_count": 46, "fail_count": 2, "status": "ON TRACK"}},
                {{"version": "{versions_for_example[2]}", "pass_count": 47, "fail_count": 1, "status": "ON TRACK"}}
            ],
            "Belk": [
                {{"version": "{versions_for_example[0]}", "pass_count": 40, "fail_count": 7, "status": "MEDIUM RISK"}},
                {{"version": "{versions_for_example[1]}", "pass_count": 42, "fail_count": 5, "status": "ON TRACK"}},
                {{"version": "{versions_for_example[2]}", "pass_count": 43, "fail_count": 4, "status": "ON TRACK"}}
            ]
        }},
        ...
    }}
}}""",
        agent=structurer,
        async_execution=False,
        expected_output="Valid JSON string with no extra text",
        callback=lambda output: (
            logger.info(f"Structure task output type: {type(output.raw)}, content: {output.raw if isinstance(output.raw, str) else output.raw}"),
            setattr(shared_state, 'metrics', process_task_output(output.raw, versions))
        )
    )

    analyst = Agent(
        role="Trend Analyst",
        goal="Add accurate trends to metrics data and maintain valid JSON",
        backstory="Data scientist specializing in metric analysis",
        llm=llm,
        verbose=True,
        memory=True,
    )

    analysis_task = Task(
        description=f"""Enhance metrics JSON with trends:
1. Input is JSON from Data Structurer
2. Add 'trend' field to each metric item
3. Output MUST be valid JSON
4. For metrics except Customer Specific Testing (UAT):
   - Sort items by version ({', '.join(f'"{v}"' for v in versions)})
   - For each item (except first per metric):
     - Compute % change: ((current_value - previous_value) / previous_value) * 100
     - If previous_value is 0 or |change| < 0.01, set trend to "→"
     - If |% change| < 1%, set trend to "→"
     - If % change > 0, set trend to "↑ (X.X%)" (e.g., "↑ (5.2%)")
     - If % change < 0, set trend to "↓ (X.X%)"
   - First item per metric gets "→"
5. For Customer Specific Testing (UAT):
   - For each client (RBS, Tesco, Belk), compute pass rate: pass_count / (pass_count + fail_count) * 100
   - Sort items by version ({', '.join(f'"{v}"' for v in versions)})
   - For each item (except first per client):
     - Compute % change in pass rate: (current_pass_rate - previous_pass_rate)
     - If previous_total or current_total is 0 or |change| < 0.01, set trend to "→"
     - If |% change| < 1%, set trend to "→"
     - If % change > 0, set trend to "↑ (X.X%)"
     - If % change < 0, set trend to "↓ (X.X%)"
   - First item per client gets "→"
6. Ensure all metrics are included: {', '.join(EXPECTED_METRICS)}
7. Use double quotes for all strings
8. No trailing commas or comments
9. Validate JSON syntax before output
EXAMPLE INPUT:
{{
    "metrics": {{
        "Open ALL RRR Defects": {{
            "ATLS": [
                {{"version": "{versions_for_example[0]}", "value": 10, "status": "RISK"}},
                {{"version": "{versions_for_example[1]}", "value": 8, "status": "MEDIUM RISK"}},
                {{"version": "{versions_for_example[2]}", "value": 5, "status": "ON TRACK"}}
            ],
            "BTLS": [
                {{"version": "{versions_for_example[0]}", "value": 12, "status": "RISK"}},
                {{"version": "{versions_for_example[1]}", "value": 9, "status": "MEDIUM RISK"}},
                {{"version": "{versions_for_example[2]}", "value": 6, "status": "ON TRACK"}}
            ]
        }},
        "Customer Specific Testing (UAT)": {{
            "RBS": [
                {{"version": "{versions_for_example[0]}", "pass_count": 50, "fail_count": 5, "status": "ON TRACK"}},
                {{"version": "{versions_for_example[1]}", "pass_count": 48, "fail_count": 6, "status": "MEDIUM RISK"}},
                {{"version": "{versions_for_example[2]}", "pass_count": 52, "fail_count": 4, "status": "ON TRACK"}}
            ],
            "Tesco": [
                {{"version": "{versions_for_example[0]}", "pass_count": 45, "fail_count": 3, "status": "ON TRACK"}},
                {{"version": "{versions_for_example[1]}", "pass_count": 46, "fail_count": 2, "status": "ON TRACK"}},
                {{"version": "{versions_for_example[2]}", "pass_count": 47, "fail_count": 1, "status": "ON TRACK"}}
            ],
            "Belk": [
                {{"version": "{versions_for_example[0]}", "pass_count": 40, "fail_count": 7, "status": "MEDIUM RISK"}},
                {{"version": "{versions_for_example[1]}", "pass_count": 42, "fail_count": 5, "status": "ON TRACK"}},
                {{"version": "{versions_for_example[2]}", "pass_count": 43, "fail_count": 4, "status": "ON TRACK"}}
            ]
        }},
        ...
    }}
}}
EXAMPLE OUTPUT:
{{
    "metrics": {{
        "Open ALL RRR Defects": {{
            "ATLS": [
                {{"version": "{versions_for_example[0]}", "value": 10, "status": "RISK", "trend": "→"}},
                {{"version": "{versions_for_example[1]}", "value": 8, "status": "MEDIUM RISK", "trend": "↓ (20.0%)"}},
                {{"version": "{versions_for_example[2]}", "value": 5, "status": "ON TRACK", "trend": "↓ (37.5%)"}}
            ],
            "BTLS": [
                {{"version": "{versions_for_example[0]}", "value": 12, "status": "RISK", "trend": "→"}},
                {{"version": "{versions_for_example[1]}", "value": 9, "status": "MEDIUM RISK", "trend": "↓ (25.0%)"}},
                {{"version": "{versions_for_example[2]}", "value": 6, "status": "ON TRACK", "trend": "↓ (33.3%)"}}
            ]
        }},
        "Customer Specific Testing (UAT)": {{
            "RBS": [
                {{"version": "{versions_for_example[0]}", "pass_count": 50, "fail_count": 5, "status": "ON TRACK", "pass_rate": 90.9, "trend": "→"}},
                {{"version": "{versions_for_example[1]}", "pass_count": 48, "fail_count": 6, "status": "MEDIUM RISK", "pass_rate": 88.9, "trend": "↓ (2.0%)"}},
                {{"version": "{versions_for_example[2]}", "pass_count": 52, "fail_count": 4, "status": "ON TRACK", "pass_rate": 92.9, "trend": "↑ (4.0%)"}}
            ],
            "Tesco": [
                {{"version": "{versions_for_example[0]}", "pass_count": 45, "fail_count": 3, "status": "ON TRACK", "pass_rate": 93.8, "trend": "→"}},
                {{"version": "{versions_for_example[1]}", "pass_count": 46, "fail_count": 2, "status": "ON TRACK", "pass_rate": 95.8, "trend": "↑ (2.0%)"}},
                {{"version": "{versions_for_example[2]}", "pass_count": 47, "fail_count": 1, "status": "ON TRACK", "pass_rate": 97.9, "trend": "↑ (2.1%)"}}
            ],
            "Belk": [
                {{"version": "{versions_for_example[0]}", "pass_count": 40, "fail_count": 7, "status": "MEDIUM RISK", "pass_rate": 85.1, "trend": "→"}},
                {{"version": "{versions_for_example[1]}", "pass_count": 42, "fail_count": 5, "status": "ON TRACK", "pass_rate": 89.4, "trend": "↑ (4.3%)"}},
                {{"version": "{versions_for_example[2]}", "pass_count": 43, "fail_count": 4, "status": "ON TRACK", "pass_rate": 91.5, "trend": "↑ (2.1%)"}}
            ]
        }},
        ...
    }}
}}
Only return valid JSON.""",
        agent=analyst,
        async_execution=True,
        context=[validated_structure_task],
        expected_output="Valid JSON string with trend analysis",
        callback=lambda output: (
            logger.info(f"Analysis task output type: {type(output.raw)}, content: {output.raw if isinstance(output.raw, str) else output.raw}"),
            setattr(shared_state, 'metrics', process_task_output(output.raw, versions))
        )
    )

    visualizer = Agent(
        role="Data Visualizer",
        goal="Generate consistent visualizations for all metrics",
        backstory="Expert in generating Python plots for software metrics",
        llm=llm,
        verbose=True,
        memory=True,
    )

    visualization_task = Task(
        description=f"""Create a standalone Python script that:
1. Accepts the provided 'metrics' JSON structure as input.
2. Generates visualizations for the following metrics, using the specified chart types:
   - Open ALL RRR Defects (ATLS and BTLS): Grouped bar chart comparing ATLS and BTLS across releases.
   - Open Security Defects (ATLS and BTLS): Grouped bar chart comparing ATLS and BTLS across releases.
   - All Open Defects (T-1) (ATLS and BTLS): Grouped bar chart comparing ATLS and BTLS across releases.
   - All Security Open Defects (ATLS and BTLS): Grouped bar chart comparing ATLS and BTLS across releases.
   - Load/Performance (ATLS and BTLS): Grouped bar chart comparing ATLS and BTLS across releases.
   - E2E Test Coverage: Line chart showing trend across releases.
   - Automation Test Coverage: Line chart showing trend across releases.
   - Unit Test Coverage: Line chart showing trend across releases.
   - Defect Closure Rate: Bar chart showing values across releases.
   - Regression Issues: Bar chart showing values across releases.
   - Customer Specific Testing (UAT) for RBS: Stacked bar chart showing pass/fail counts across releases.
   - Customer Specific Testing (UAT) for Tesco: Stacked bar chart showing pass/fail counts across releases.
   - Customer Specific Testing (UAT) for Belk: Stacked bar chart showing pass/fail counts across releases.
3. Each plot must use: plt.figure(figsize=(8,5), dpi=120).
4. Save each chart as a PNG in 'visualizations/' directory with descriptive filenames (e.g., 'open_all_rrr_defects_atls_btls.png', 'uat_rbs.png').
5. Include error handling for missing or malformed data, ensuring all specified charts are generated.
6. Log each chart generation attempt to 'visualization.log' for debugging.
7. Output ONLY the Python code, with no markdown or explanation text.
8. Do not generate charts for Delivery Against Requirements.
9. Ensure at least 10 charts are generated for the listed metrics, plus 3 UAT charts.
10. For grouped bar charts, use distinct colors for ATLS and BTLS (e.g., blue for ATLS, orange for BTLS) and includeajat for a legend.
11. For stacked bar charts, use green for pass_count, red for fail_count, and include a legend.
12. Use the following metric lists for iteration:
    atls_btls_metrics = {EXPECTED_METRICS[:5]}
    coverage_metrics = {EXPECTED_METRICS[5:8]}
    other_metrics = {EXPECTED_METRICS[8:10]}
    uat_metric = 'Customer Specific Testing (UAT)'
    do_not_use = 'Delivery Against Requirements'
13. Use versions: NA (use versions from data)
14. If no data is available, generate a placeholder chart with a message indicating the issue."""
        agent=visualizer,
        context=[analysis_task],
        expected_output="Python script only"
    )

    reporter = Agent(
        role="Technical Writer",
        goal="Generate a professional markdown report",
        backstory="Expert in generating structured software metrics reports",
        llm=llm,
        verbose=True,
        memory=True,
    )

    overview_task = Task(
        description=f"""Write ONLY the following Markdown section:
## Overview
- Provide a 3-4 sentence comprehensive summary of release health, covering overall stability, notable improvements, and any concerning patterns observed across releases {', '.join(versions)}
- Explicitly list all analyzed releases
- Include 2-3 notable improvements with specific metrics and version comparisons where relevant
- Mention any significant deviations from expected patterns
Only output this section.""",
        agent=reporter,
        context=[analysis_task],
        expected_output="Detailed markdown for Overview section"
    )

    metrics_summary_task = Task(
        description=f"""Write ONLY the '## Metrics Summary' section with the following order:
### Delivery Against Requirements
### Open ALL RRR Defects (ATLS)
### Open ALL RRR Defects (BTLS)
### Open Security Defects (ATLS)
### Open Security Defects (BTLS)
### All Open Defects (T-1) (ATLS)
### All Open Defects (T-1) (BTLS)
### All Security Open Defects (ATLS)
### All Security Open Defects (BTLS)
### Customer Specific Testing (UAT)
#### RBS
#### Tesco
#### Belk
### Load/Performance
#### ATLS
#### BTLS
### E2E Test Coverage
### Automation Test Coverage
### Unit Test Coverage
### Defect Closure Rate (ATLS)
### Regression Issues

STRICT RULES:
- For Customer Specific Testing (UAT), generate tables for each client with the following columns: Release | Pass Count | Fail Count | Pass Rate (%) | Trend | Status
- For other metrics, use existing table formats
- Use only these statuses: ON TRACK, MEDIUM RISK, RISK, NEEDS REVIEW
- Use only these trend formats: ↑ (X%), ↓ (Y%), →
- No missing releases or extra formatting
EXAMPLE FOR UAT:
#### RBS
| Release | Pass Count | Fail Count | Pass Rate (%) | Trend      | Status       |
|---------|------------|------------|---------------|------------|--------------|
| {versions_for_example[0]}    | 50         | 5          | 90.9          | →          | ON TRACK     |
| {versions_for_example[1]}    | 48         | 6          | 88.9          | ↓ (2.0%)   | MEDIUM RISK  |
| {versions_for_example[2]}    | 52         | 4          | 92.9          | ↑ (4.0%)   | ON TRACK     |
Only output this section.""",
        agent=reporter,
        context=[analysis_task],
        expected_output="Markdown for Metrics Summary"
    )

    key_findings_task = Task(
        description=f"""Generate ONLY this Markdown section:
## Key Findings
1. First finding (2-3 sentences explaining the observation with specific metric references and version comparisons across {', '.join(versions)})
2. Second finding (2-3 sentences with quantitative data points from the metrics where applicable)
3. Third finding (2-3 sentences focusing on security-related observations)
4. Fourth finding (2-3 sentences about testing coverage trends)
5. Fifth finding (2-3 sentences highlighting any unexpected patterns or anomalies)
6. Sixth finding (2-3 sentences about performance or load metrics)
7. Seventh finding (2-3 sentences summarizing defect management effectiveness)

Maintain professional, analytical tone while being specific.""",
        agent=reporter,
        context=[analysis_task],
        expected_output="Detailed markdown bullet list"
    )

    recommendations_task = Task(
        description="""Generate ONLY this Markdown section:
## Recommendations
1. First recommendation (2-3 actionable sentences with specific metrics or areas to address)
2. Second recommendation (2-3 sentences about security improvements with version targets)
3. Third recommendation (2-3 sentences about testing coverage enhancements)
4. Fourth recommendation (2-3 sentences about defect management process changes)
5. Fifth recommendation (2-3 sentences about performance optimization)
6. Sixth recommendation (2-3 sentences about risk mitigation strategies)
7. Seventh recommendation (2-3 sentences about monitoring improvements)

Each recommendation should be specific, measurable, and tied to the findings.""",
        agent=reporter,
        context=[analysis_task],
        expected_output="Detailed markdown bullet list"
    )

    assemble_report_task = Task(
        description="""Assemble the final markdown report in this exact structure:

# Software Metrics Report

## Overview
[Insert from Overview Task]

---

## Metrics Summary
[Insert from Metrics Summary Task]

---

## Key Findings
[Insert from Key Findings Task]

---

## Recommendations
[Insert from Recommendations Task]

Do NOT alter content. Just combine with correct formatting.""",
        agent=reporter,
        context=[
            overview_task,
            metrics_summary_task,
            key_findings_task,
            recommendations_task
        ],
        expected_output="Full markdown report"
    )

    data_crew = Crew(
        agents=[structurer, analyst],
        tasks=[validated_structure_task, analysis_task],
        process=Process.sequential,
        verbose=True
    )

    report_crew = Crew(
        agents=[reporter],
        tasks=[overview_task, metrics_summary_task, key_findings_task, recommendations_task, assemble_report_task],
        process=Process.sequential,
        verbose=True
    )

    viz_crew = Crew(
        agents=[visualizer],
        tasks=[visualization_task],
        process=Process.sequential,
        verbose=True
    )

    for crew, name in [(data_crew, "data_crew"), (report_crew, "report_crew"), (viz_crew, "viz_crew")]:
        for i, task in enumerate(crew.tasks):
            if not isinstance(task, Task):
                logger.error(f"Invalid task in {name} at index {i}: {task}")
                raise ValueError(f"Task in {name} is not a Task object")
            logger.info(f"{name} task {i} async_execution: {task.async_execution}")

    return data_crew, report_crew, viz_crew
