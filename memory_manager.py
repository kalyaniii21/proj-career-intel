import json
import os

from dotenv import load_dotenv
from upstash_redis import Redis


# Load connection variables from your .env file.
load_dotenv()


class UpstashSessionMemory:
    def __init__(self):
        url = os.getenv("UPSTASH_REDIS_REST_URL")
        token = os.getenv("UPSTASH_REDIS_REST_TOKEN")

        if not url or not token:
            raise ValueError("Missing Upstash Redis credentials in .env file!")

        self.redis = Redis(url=url, token=token)
        print("Secure connection to Upstash Redis Memory Layer initialized.")

    def get_session_history(self, session_id: str) -> list:
        """Retrieve the structured conversation history for a user session."""
        key = f"session:{session_id}:history"
        raw_data = self.redis.get(key)

        if raw_data:
            return json.loads(raw_data)
        return []

    def add_message_to_session(self, session_id: str, role: str, message: str, max_history: int = 10):
        """
        Append a turn to the session history and keep only the latest entries.
        """
        key = f"session:{session_id}:history"
        history = self.get_session_history(session_id)

        history.append({
            "role": role,
            "content": message,
        })

        if len(history) > max_history:
            print(f"Session {session_id} history exceeded limit. Truncating oldest turns.")
            history = history[-max_history:]

        self.redis.set(key, json.dumps(history), ex=86400)
        print(f"Successfully persisted turn for '{role}' in session: {session_id}")

    def clear_session(self, session_id: str):
        """Reset a session when a student restarts their interview track."""
        key = f"session:{session_id}:history"
        self.redis.delete(key)
        print(f"Session memory wiped for: {session_id}")
