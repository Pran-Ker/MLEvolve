import logging
from . import gemini as _gemini
from . import claude as _claude
from .gemini import FunctionSpec, OutputType, PromptType, compile_prompt_to_md
from config import Config
logger = logging.getLogger("MLEvolve")


def _is_claude(model: str) -> bool:
    return model.startswith("claude")


def query(
    system_message: PromptType | None,
    user_message: PromptType | None,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    func_spec: FunctionSpec | None = None,
    cfg:Config=None,
    **model_kwargs,
) -> OutputType:
    """
    General LLM query for various backends with a single system and user message.
    Supports function calling for some backends.

    Args:
        system_message (PromptType | None): Uncompiled system message (will generate a message following the OpenAI/Anthropic format)
        user_message (PromptType | None): Uncompiled user message (will generate a message following the OpenAI/Anthropic format)
        model (str): string identifier for the model to use (e.g. "gemini-3-pro-preview")
        temperature (float | None, optional): Temperature to sample at. Defaults to the model-specific default.
        max_tokens (int | None, optional): Maximum number of tokens to generate. Defaults to the model-specific max tokens.
        func_spec (FunctionSpec | None, optional): Optional FunctionSpec object defining a function call. If given, the return value will be a dict.

    Returns:
        OutputType: A string completion if func_spec is None, otherwise a dict with the function call details.
    """

    model_kwargs = model_kwargs | {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    logger.info("---Querying model---", extra={"verbose": True})
    system_message = compile_prompt_to_md(system_message) if system_message else None
    if system_message:
        if len(system_message) > 1000:
            logger.info(f"system: {system_message[-1000:]}", extra={"verbose": True})
        else:
            logger.info(f"system: {system_message}", extra={"verbose": True})
    user_message = compile_prompt_to_md(user_message) if user_message else None
    if user_message:
        if len(user_message) > 1000:
            logger.info(f"user: {user_message[-1000:]}", extra={"verbose": True})
        else:
            logger.info(f"user: {user_message}", extra={"verbose": True})
    if func_spec:
        logger.info(f"function spec: {func_spec.to_dict()}", extra={"verbose": True})

    backend = _claude if _is_claude(model) else _gemini

    output, req_time, in_tok_count, out_tok_count, info = backend.query(
        system_message=system_message,
        user_message=user_message,
        func_spec=func_spec,
        cfg=cfg,
        **model_kwargs,
    )
    logger.info("---Query complete---", extra={"verbose": True})

    return output


def chat(
    messages: list[dict],
    system_message: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    cfg: Config = None,
) -> str:
    """Multi-turn chat dispatched to the correct backend based on model name."""
    if model is None:
        model = cfg.agent.code.model
    backend = _claude if _is_claude(model) else _gemini
    return backend.chat(
        messages=messages,
        system_message=system_message,
        cfg=cfg,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


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
    """Agentic chat with tool-use loop (Claude backend only)."""
    return _claude.agentic_chat(
        messages=messages,
        system_message=system_message,
        tools=tools,
        tool_executor=tool_executor,
        cfg=cfg,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_tool_rounds=max_tool_rounds,
    )


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
    """Dispatch generate() to the correct backend based on model name."""
    model = cfg.agent.code.model
    backend = _claude if _is_claude(model) else _gemini
    return backend.generate(
        prompt=prompt,
        cfg=cfg,
        temperature=temperature,
        max_tokens=max_tokens,
        stop_tokens=stop_tokens,
        json_schema=json_schema,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )
