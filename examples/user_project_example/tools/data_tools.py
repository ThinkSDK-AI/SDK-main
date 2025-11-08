"""
Example tools module

Tools can be defined in two ways:
1. Using the @tool decorator (recommended)
2. Using the tool_ prefix
3. Exporting via __tools__ dict
"""

from fourier.config import tool
import json
from typing import Dict, Any


@tool
def fetch_user_data(user_id: str) -> Dict[str, Any]:
    """
    Fetch user data from database (mock implementation).

    Args:
        user_id: User ID to fetch

    Returns:
        User data dictionary
    """
    # Mock implementation
    return {
        "user_id": user_id,
        "name": "John Doe",
        "email": f"user{user_id}@example.com",
        "status": "active"
    }


@tool
def send_notification(user_id: str, message: str, channel: str = "email") -> bool:
    """
    Send notification to user.

    Args:
        user_id: Target user ID
        message: Notification message
        channel: Channel to use (email, sms, push)

    Returns:
        Success status
    """
    print(f"Sending {channel} notification to {user_id}: {message}")
    return True


# Alternative: using tool_ prefix
def tool_calculate_metrics(data: Dict[str, Any]) -> Dict[str, float]:
    """Calculate metrics from data"""
    return {
        "total": sum(data.values()),
        "average": sum(data.values()) / len(data) if data else 0,
        "count": len(data)
    }


# Alternative: explicit export via __tools__
def custom_formatter(data: Any) -> str:
    """Custom data formatter"""
    return json.dumps(data, indent=2)


__tools__ = {
    "format_json": custom_formatter
}
