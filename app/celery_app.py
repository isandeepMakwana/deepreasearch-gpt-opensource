import os
import boto3
from celery import Celery

# AWS Region configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
SQS_QUEUE_NAME = os.getenv("SQS_QUEUE_NAME", "deep-research-queue")

# Redis remains as the result backend
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Create SQS client using EC2 instance role
sqs = boto3.client("sqs", region_name=AWS_REGION)

# Optionally create the queue if it doesn't exist
try:
    response = sqs.create_queue(QueueName=SQS_QUEUE_NAME)
    queue_url = response["QueueUrl"]
    print(f"SQS Queue URL: {queue_url}")
except Exception as e:
    print(f"Error creating/accessing SQS queue: {e}")

# Use IAM role for authentication with SQS
celery_app = Celery(
    "deep_research",
    broker=f"sqs://",
    backend=REDIS_URL,
)

# Configure Celery to use SQS with optimized settings
celery_app.conf.update(
    task_track_started=True,
    result_expires=3600,  # 1 hour
    task_compression='gzip',  # Enable gzip compression
    task_serializer='json',  # Use JSON serializer
    accept_content=['json'],  # Accept JSON content
    result_serializer='json',  # Use JSON for results
    result_backend_transport_options={
        'retry_policy': {
            'timeout': 5.0
        }
    },
    broker_transport_options={
        "region": AWS_REGION,
        "queue_name_prefix": "",
        "visibility_timeout": 3600,  # 1 hour
        "max_retries": 3,  # Number of retries for failed tasks
        "compression_level": 9,  # Maximum compression level
    },
)

from . import tasks
