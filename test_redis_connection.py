import os

import redis
from dotenv import load_dotenv

load_dotenv()

client = redis.Redis(
    host=os.environ["REDIS_HOST"],
    port=int(os.environ["REDIS_PORT"]),
    username=os.environ["REDIS_USERNAME"],
    password=os.environ["REDIS_PASSWORD"],
    ssl=False,
)

test_key = "connection_test"
test_value = "hello from search-intent-classifier"

client.set(test_key, test_value)
result = client.get(test_key).decode()

print(f"Got back: {result}")

if result == test_value:
    print("Connection successful!")
