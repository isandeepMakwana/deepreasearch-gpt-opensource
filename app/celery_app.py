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
