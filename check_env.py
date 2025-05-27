import os

redis_url = os.getenv("REDIS_URL", "Not set")
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID", "Not set")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "Not set")
aws_region = os.getenv("AWS_REGION", "Not set")

print(f"REDIS_URL: {redis_url}")
print(f"AWS_ACCESS_KEY_ID: {'*****' if aws_access_key != 'Not set' else 'Not set'}")
print(f"AWS_SECRET_ACCESS_KEY: {'*****' if aws_secret_key != 'Not set' else 'Not set'}")
print(f"AWS_REGION: {aws_region}")
