import os
import boto3
from celery import Celery

# AWS SQS configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
SQS_QUEUE_NAME = os.getenv("SQS_QUEUE_NAME", "deep-research-queue")

# Redis remains as the result backend
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Create SQS client to ensure queue exists
if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
    sqs = boto3.client(
        "sqs",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )

    # Optionally create the queue if it doesn't exist
    try:
        response = sqs.create_queue(QueueName=SQS_QUEUE_NAME)
        queue_url = response["QueueUrl"]
        print(f"SQS Queue URL: {queue_url}")
    except Exception as e:
        print(f"Error creating/accessing SQS queue: {e}")

# SQS broker URL format
SQS_BROKER_URL = f"sqs://{AWS_ACCESS_KEY_ID}:{AWS_SECRET_ACCESS_KEY}@"

celery_app = Celery(
    "deep_research",
    broker=SQS_BROKER_URL,
    backend=REDIS_URL,
)

# Configure Celery to use SQS
celery_app.conf.update(
    task_track_started=True,
    result_expires=3600,  # 1 hour
    broker_transport_options={
        "region": AWS_REGION,
        "queue_name_prefix": "",
        "visibility_timeout": 3600,  # 1 hour
    },
)

from . import tasks
