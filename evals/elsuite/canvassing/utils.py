import os
import re
from typing import Literal, Dict

from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def is_system_msg(m: dict) -> bool:
    assert isinstance(m, dict), "Message must be a dict."
    assert "role" in m, "Message must have a role."
    assert isinstance(m["role"], str), "Message role must be a string."

    return "system" == m["role"]


def format_msg(msg: dict, role: Literal["mark", "voter", "canvasser", "system"]) -> Dict[str, str]:
    """Formats a single message from the perspective of the `role`."""
    # body
    is_others_msg = role not in msg["role"]
    if role in {"mark", "voter"} and msg["role"] in {"mark", "voter"}:
        is_others_msg = False
    new_content = msg["content"]

    
    if msg["role"] == "system":
        new_role = "system"
    elif is_others_msg:
        new_role = "user"
    else:
        new_role = "assistant"

    new_message = {
        "content": new_content,
        "role": new_role,
    }

    # post-conditions
    assert isinstance(new_message["content"], str)
    assert new_message["role"] in ["user", "assistant", "system"]

    return new_message


def format_msgs(
    messages: list[dict],
    role: Literal["mark", "canvasser"],
) -> list[dict]:
    """Format messages from the perspective of the `role`."""
    new_messages = [format_msg(msg, role) for msg in messages]

    # post-conditions
    for m in new_messages:
        assert m["role"] in ["user", "assistant", "system"]

    return new_messages


def model_broke_character(message: str) -> bool:
    return (
        "AI language model" in message.split("[OPEN]")[1]
        if "[OPEN]" in message
        else "AI language model" in message
    )


def model_output_empty_tags(message: str) -> bool:
    pattern = r"\[OPEN\]\s*$"
    return bool(re.search(pattern, message))


def openai_chatcompletion_create(*args, **kwargs):
    return client.chat.completions.create(*args, **kwargs)
