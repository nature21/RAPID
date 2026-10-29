import argparse
from enum import Enum
import json
import os
import re
import time

import numpy as np
from openai import OpenAI
from PIL import Image

from real2sim.utils import convert_pil_image_to_base64

PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")


class VLMBackend(str, Enum):
    OPENAI = "OPENAI"
    CUSTOM = "CUSTOM"


DEFAULT_BACKEND = VLMBackend.CUSTOM
BACKEND_ENV_VAR = "REAL2SIM_VLM_BACKEND"

BACKEND_CONFIGS = {
    VLMBackend.OPENAI: {
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "stream": False,
    },
    VLMBackend.CUSTOM: {
        "api_key_env": "CUSTOM_OPENAI_API_KEY",
        "base_url_env": "CUSTOM_OPENAI_BASE_URL",
        "stream": False,
    },
}


def resolve_backend(backend=None):
    value = backend or os.getenv(BACKEND_ENV_VAR, DEFAULT_BACKEND.value)
    if isinstance(value, VLMBackend):
        return value
    try:
        return VLMBackend(value.upper())
    except (AttributeError, ValueError) as exc:
        choices = ", ".join(item.value for item in VLMBackend)
        raise ValueError(f"Unknown VLM backend {value!r}; choose one of: {choices}") from exc


def get_backend_config(backend=None):
    selected_backend = resolve_backend(backend)
    config = BACKEND_CONFIGS[selected_backend]

    api_key_env = config["api_key_env"]
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(
            f"{api_key_env} is not set for VLM backend {selected_backend.value}"
        )

    base_url_env = config.get("base_url_env")
    base_url = os.getenv(base_url_env) if base_url_env else config.get("base_url")
    if not base_url:
        raise ValueError(
            f"{base_url_env} is not set for VLM backend {selected_backend.value}"
        )

    return selected_backend, api_key, base_url, config["stream"]


def load_prompts(prompt_dir):
    """Load prompts from files.
    """
    prompts = dict()
    for filename in os.listdir(prompt_dir):
        path = os.path.join(prompt_dir, filename)
        if os.path.isfile(path) and path[-4:] == '.txt':
            with open(path, 'r') as f:
                value = f.read()
            key = filename[:-4]
            prompts[key] = value
    return prompts

def parse_selection_results(text):
    """Parse a VLM JSON result while tolerating common response wrappers."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Selection results response is empty")

    candidates = []
    wrapper_patterns = (
        r"\*\*\*\s*(.*?)\s*\*\*\*",
        r"<result\b[^>]*>\s*(.*?)\s*</result>",
        r"```(?:json)?\s*(.*?)\s*```",
    )
    for pattern in wrapper_patterns:
        candidates.extend(
            match.group(1).strip()
            for match in re.finditer(pattern, text, re.S | re.I)
            if match.group(1).strip()
        )

    stripped = text.strip()
    candidates.append(stripped)

    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        for match in re.finditer(r"[\{\[]", candidate):
            try:
                parsed, _ = decoder.raw_decode(candidate[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, (dict, list)):
                return parsed

    raise ValueError("No valid JSON selection result found")

def prepare_inputs(
        messages,
        images=None,
        meta_prompt=None,
        example_images=None,
        example_responses=None,
):
    if not isinstance(messages, list):
        messages = [messages]

    user_content = []
    system_content = []

    for message in messages:
        content = {
            'type': 'text',
            'text': message
        }
        user_content.append(content)

    if images is not None:
        if not isinstance(images, list):
            images = [images]

        for img in images:
            b64 = convert_pil_image_to_base64(img)

            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}"
                }
            })

    if meta_prompt is not None:
        system_content = [{
            "type": "text",
            "text": meta_prompt
        }]
    else:
        system_content = []

    if example_images is not None and example_responses is not None:
        if not isinstance(example_images, list):
            example_images = [example_images]
        if not isinstance(example_responses, list):
            example_responses = [example_responses]

        for example_image, example_response in zip(
                example_images, example_responses):
            b64 = convert_pil_image_to_base64(example_image)

            content = {
                'type': 'image_url',
                'image_url': {
                    'url': f'data:image/jpeg;base64,{b64}'
                }
            }
            user_content.append(content)

            content = {
                'type': 'text',
                'text': example_response,
            }
            user_content.append(content)

    messages = []
    if system_content:
        messages.append({
            "role": "system",
            "content": system_content
        })
    messages.append({
        "role": "user",
        "content": user_content
    })
    return messages

def prepare_inputs_fast(
        messages,
        images=None,
        meta_prompt=None,
        example_images=None,
        example_responses=None,
):
    prepared = []

    if meta_prompt:
        prepared.append({
            "role": "system",
            "content": [{"type": "text", "text": meta_prompt}]
        })

    if not isinstance(messages, list):
        messages = [messages]

    for msg in messages:
        prepared.append({
            "role": "user",
            "content": [{"type": "text", "text": msg}]
        })

    if example_images and example_responses:
        if not isinstance(example_images, list):
            example_images = [example_images]
        if not isinstance(example_responses, list):
            example_responses = [example_responses]

        for img, resp in zip(example_images, example_responses):
            b64 = convert_pil_image_to_base64(img)

            prepared.append({
                "role": "user",
                "content": [{
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}"
                    }
                }]
            })

            prepared.append({
                "role": "user",
                "content": [{"type": "text", "text": resp}]
            })

    if images:
        if not isinstance(images, list):
            images = [images]

        for img in images:
            b64 = convert_pil_image_to_base64(img)
            prepared.append({
                "role": "user",
                "content": [{
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}"
                    }
                }]
            })

    return prepared

def query_model(messages,
                selected_model,
                images=None,
                meta_prompt=None,
                example_images=None,
                example_responses=None,
                backend=None,
                reasoning_effort=None,
        ):
    _, api_key, base_url, stream = get_backend_config(backend)
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    prepared_messages = prepare_inputs(
        messages,
        images,
        meta_prompt,
        example_images,
        example_responses
    )

    request = {
        "model": selected_model,
        "messages": prepared_messages,
    }
    if reasoning_effort is not None:
        request["reasoning_effort"] = reasoning_effort

    if stream:
        response = client.chat.completions.create(stream=True, **request)

        content_parts = []
        for chunk in response:
            for choice in chunk.choices:
                content = getattr(choice.delta, "content", None)
                if content:
                    content_parts.append(content)
        return "".join(content_parts)

    response = client.chat.completions.create(**request)

    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser(description="Test real2sim VLM query_model.")
    parser.add_argument(
        "--backend",
        type=str.upper,
        choices=[backend.value for backend in VLMBackend],
        help=(
            f"VLM API backend. Defaults to ${BACKEND_ENV_VAR}, or "
            f"{DEFAULT_BACKEND.value} when unset."
        ),
    )
    parser.add_argument(
        "--model",
        default="gpt-6-astra",
        help="Model name to send to the selected backend.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high"),
        default="high",
        help="Reasoning effort for the VLM request; 'none' omits the parameter.",
    )
    parser.add_argument(
        "-m",
        "--message",
        action="append",
        help="User message to send. Can be passed multiple times.",
    )
    parser.add_argument(
        "--prompt",
        help="Name of a .txt prompt from PROMPT_DIR, without the .txt suffix.",
    )
    parser.add_argument(
        "-i",
        "--image",
        action="append",
        help="Path to an image to include. Can be passed multiple times.",
    )
    parser.add_argument(
        "--meta-prompt",
        help="Optional system prompt.",
    )
    args = parser.parse_args()

    prompts = load_prompts(PROMPT_DIR)
    messages = []

    if args.prompt:
        if args.prompt not in prompts:
            available = ", ".join(sorted(prompts)) or "none"
            parser.error(f"Unknown prompt '{args.prompt}'. Available prompts: {available}")
        prompt_text = prompts[args.prompt].strip()
        if prompt_text:
            messages.append(prompt_text)

    if args.message:
        messages.extend(args.message)

    if not messages:
        available = ", ".join(sorted(prompts)) or "none"
        parser.error(
            "Provide --message and/or --prompt. "
            f"Available prompts in {PROMPT_DIR}: {available}"
        )

    try:
        get_backend_config(args.backend)
    except ValueError as exc:
        parser.error(str(exc))

    images = None
    if args.image:
        images = [Image.open(path).convert("RGB") for path in args.image]

    response = query_model(
        messages=messages,
        selected_model=args.model,
        images=images,
        meta_prompt=args.meta_prompt,
        backend=args.backend,
        reasoning_effort=(
            None if args.reasoning_effort == "none" else args.reasoning_effort
        ),
    )
    print(response)

if __name__ == "__main__":
    main()
