import os
from celery import Celery

# Default to localhost settings; adjust if needed.
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672//")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "deep_research",
    broker=RABBITMQ_URL,
    backend=REDIS_URL,
)

# Optional: Celery configuration, e.g. concurrency, time limits, etc.
celery_app.conf.update(
    task_track_started=True,
    result_expires=3600,  # 1 hour
)

from . import tasks