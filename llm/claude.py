"""Anthropic Claude API backend: function calling (query), streaming generation (generate),
   retry logic, and structured output via tool_use."""

import logging
import time

from funcy import notnone, once, select_values
from config import Config

logger = logging.getLogger("MLEvolve")

_client = None

CLAUDE_TIMEOUT_EXCEPTIONS = (Exception,)


@once
def _setup_claude_client(cfg: Config):
    global _client
    import anthropic
    kwargs = {"api_key": cfg.agent.code.api_key} if cfg.agent.code.api_key else {}
    if cfg.agent.code.base_url:
        kwargs["base_url"] = cfg.agent.code.base_url
    _client = anthropic.Anthropic(**kwargs)


def _func_spec_to_tool(func_spec):
    """Convert a FunctionSpec to Claude tool format."""
    return {
        "name": func_spec.name,
        "description": func_spec.description,
        "input_schema": func_spec.json_schema,
    }


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec=None,
    cfg: Config = None,
    **model_kwargs,
) -> tuple:
    _setup_claude_client(cfg)
    filtered_kwargs: dict = select_values(notnone, model_kwargs)

    model = filtered_kwargs.get("model", "claude-sonnet-4-20250514")
    temperature = filtered_kwargs.get("temperature", 1.0)
    max_tokens = filtered_kwargs.get("max_tokens", 16384)

    messages = []
    # Claude requires at least one user message
    content = ""
    if user_message:
        content = user_message
    elif system_message and not user_message:
        # If only system message, put it as user message and clear system
        content = system_message
        system_message = None
    if not content:
        raise ValueError("Either system_message or user_message must be provided")

    messages.append({"role": "user", "content": content})

    api_kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }
    if system_message:
        api_kwargs["system"] = system_message

    if func_spec is not None:
        tool = _func_spec_to_tool(func_spec)
        api_kwargs["tools"] = [tool]
        api_kwargs["tool_choice"] = {"type": "tool", "name": func_spec.name}

    t0 = time.time()
    logger.info(f"Querying Claude with model: {model}")

    response = _client.messages.create(**api_kwargs)
    req_time = time.time() - t0

    # Parse response
    if func_spec is None:
        output = ""
        for block in response.content:
            if block.type == "text":
                output += block.text
        logger.info(f"Claude response: {output}", extra={"verbose": True})
    else:
        output = None
        for block in response.content:
            if block.type == "tool_use":
                output = block.input
                break
        if output is None:
            raise ValueError("No tool_use block in Claude response for structured output")
        logger.info(f"Claude structured output response: {output}", extra={"verbose": True})

    in_tokens = response.usage.input_tokens
    out_tokens = response.usage.output_tokens
    info = {"model": model, "created": int(time.time())}

    return output, req_time, in_tokens, out_tokens, info


def chat(
    messages: list[dict],
    system_message: str | None = None,
    cfg: Config = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Multi-turn chat with full message history.

    Args:
        messages: List of {"role": "user"|"assistant", "content": str} dicts.
        system_message: Optional system prompt.
        cfg: Config for client setup.
        model: Model identifier.
        temperature: Sampling temperature.
        max_tokens: Max output tokens.

    Returns:
        The assistant's text response.
    """
    _setup_claude_client(cfg)

    api_kwargs = {
        "model": model or "claude-sonnet-4-20250514",
        "max_tokens": max_tokens or 16384,
        "temperature": temperature if temperature is not None else 1.0,
        "messages": messages,
    }
    if system_message:
        api_kwargs["system"] = system_message

    response = _client.messages.create(**api_kwargs)

    output = ""
    for block in response.content:
        if block.type == "text":
            output += block.text
    return output


def agentic_chat(
    messages: list[dict],
    system_message: str | None = None,
    tools: list[dict] | None = None,
    tool_executor=None,
    cfg: Config = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_tool_rounds: int = 5,
) -> tuple[str, list[dict]]:
    """Multi-turn chat with tool-use loop.

    Calls messages.create() with tools. If response has tool_use blocks,
    executes via tool_executor(name, input) -> str, appends tool_result,
    re-calls until text-only or max_tool_rounds exhausted.

    Returns (text_response, final_messages).
    """
    _setup_claude_client(cfg)

    model = model or "claude-sonnet-4-20250514"
    max_tokens = max_tokens or 16384
    temperature = temperature if temperature is not None else 1.0

    working_messages = list(messages)
    text_parts = []

    for round_idx in range(max_tool_rounds):
        api_kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": working_messages,
        }
        if system_message:
            api_kwargs["system"] = system_message
        if tools:
            api_kwargs["tools"] = tools

        logger.info(f"[agentic_chat] round {round_idx + 1}/{max_tool_rounds}, model={model}")
        response = _client.messages.create(**api_kwargs)

        text_parts = []
        tool_uses = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_uses.append(block)

        working_messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use" or not tool_uses or not tool_executor:
            break

        logger.info(f"[agentic_chat] executing {len(tool_uses)} tool call(s)")
        tool_results = []
        for tool_use in tool_uses:
            result = tool_executor(tool_use.name, tool_use.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": str(result),
            })
        working_messages.append({"role": "user", "content": tool_results})

    return "".join(text_parts), working_messages


def generate(
    prompt: str | dict | list,
    cfg: Config,
    temperature: float | None = None,
    max_tokens: int | None = None,
    stop_tokens: list[str] | None = None,
    json_schema: dict | None = None,
    max_retries: int = 20,
    retry_delay: float = 3,
) -> str:
    """Streaming text generation via Claude API."""
    _setup_claude_client(cfg)

    from .gemini import compile_prompt_to_md

    if prompt is not None and not isinstance(prompt, str):
        prompt = compile_prompt_to_md(prompt)

    logger.info(f"generate prompt: {prompt}", extra={"verbose": True})

    model_name = cfg.agent.code.model

    for attempt in range(max_retries):
        api_kwargs = {
            "model": model_name,
            "max_tokens": max_tokens if max_tokens is not None else 16384,
            "temperature": temperature if temperature is not None else 1.0,
            "messages": [{"role": "user", "content": prompt}],
        }
        if stop_tokens:
            api_kwargs["stop_sequences"] = stop_tokens

        if json_schema is not None:
            # Use prefill technique for JSON output
            api_kwargs["messages"].append({"role": "assistant", "content": "{"})
            logger.info("Requesting JSON output from Claude", extra={"verbose": True})

        try:
            full_text = ""
            with _client.messages.stream(**api_kwargs) as stream:
                for text in stream.text_stream:
                    full_text += text

            if json_schema is not None:
                full_text = "{" + full_text

            logger.info(f"generate response: {full_text}", extra={"verbose": True})
            return full_text

        except Exception as e:
            logger.warning(f"generate failed, retrying {attempt + 1}/{max_retries}: {e}")
            if attempt >= max_retries - 1:
                logger.error("generate retry limit reached")
                raise
            time.sleep(retry_delay)
