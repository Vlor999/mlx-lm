# Copyright © 2023-2024 Apple Inc.

import argparse
import os
from atexit import register

import mlx.core as mx

from .generate import stream_generate
from .models.cache import make_prompt_cache
from .sample_utils import make_sampler
from .utils import load, sharded_load

DEFAULT_TEMP = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_XTC_PROBABILITY = 0.0
DEFAULT_XTC_THRESHOLD = 0.0
DEFAULT_SEED = 0
DEFAULT_MAX_TOKENS = 256
DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
DEFAULT_HISTORY_FILE = os.path.expanduser("~/.mlx_lm_chat_history")
DEFAULT_HISTORY_SIZE = 1000


def setup_readline(history_file: str, history_size: int):
    """Set up readline with persistent history."""
    try:
        from readline import read_history_file, set_history_length, write_history_file
    except ImportError:
        print(
            "[WARNING] readline not available. "
            "Install pyreadline3 on Windows for history support."
        )
        return

    set_history_length(history_size)

    try:
        if os.path.exists(history_file):
            read_history_file(history_file)
    except (IOError, OSError):
        pass  # History file doesn't exist or can't be read

    register(write_history_file, history_file)
    print(f"[INFO] Writting into the file : {os.path.abspath(history_file)}")


def clear_history(history_file: str):
    if os.path.exists(history_file):
        with open(history_file, "w") as f:
            return
    print(
        f"[WARNING] The file ({os.path.abspath(history_file)}) does not exists - not able to clear history"
    )


def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(description="Chat with an LLM")
    parser.add_argument(
        "--model",
        type=str,
        help="The path to the local model directory or Hugging Face repo.",
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config.",
    )
    parser.add_argument(
        "--temp", type=float, default=DEFAULT_TEMP, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=DEFAULT_TOP_P, help="Sampling top-p"
    )
    parser.add_argument(
        "--xtc-probability",
        type=float,
        default=DEFAULT_XTC_PROBABILITY,
        help="Probability of XTC sampling to happen each next token",
    )
    parser.add_argument(
        "--xtc-threshold",
        type=float,
        default=0.0,
        help="Thresold the probs of each next token candidate to be sampled by XTC",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="PRNG seed",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        help="Set the maximum key-value cache size",
        default=None,
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="System prompt to be used for the chat template",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Use pipelining instead of tensor parallelism",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Disable persistent command history",
    )
    parser.add_argument(
        "--history-file",
        type=str,
        default=DEFAULT_HISTORY_FILE,
        help=f"Path to the history file (default: {DEFAULT_HISTORY_FILE})",
    )
    parser.add_argument(
        "--history-size",
        type=int,
        default=DEFAULT_HISTORY_SIZE,
        help=f"Maximum number of history entries to save (default: {DEFAULT_HISTORY_SIZE})",
    )
    parser.add_argument(
        "--clear-history",
        action="store_true",
        help="Clear the history file",
    )
    return parser


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    group = mx.distributed.init()
    rank = group.rank()
    pipeline_group = group if args.pipeline else None
    tensor_group = group if not args.pipeline else None

    if args.clear_history and rank == 0:
        clear_history(args.history_file)

    if not args.no_history and rank == 0:
        setup_readline(args.history_file, args.history_size)

    def rprint(*args, **kwargs):
        if rank == 0:
            print(*args, **kwargs)

    mx.random.seed(args.seed)

    if group.size() > 1:
        if args.adapter_path:
            parser.error("Adapters not supported in distributed mode")
        model, tokenizer = sharded_load(args.model, pipeline_group, tensor_group)
    else:
        model, tokenizer = load(
            args.model,
            adapter_path=args.adapter_path,
            tokenizer_config={
                "trust_remote_code": True if args.trust_remote_code else None
            },
        )

    def print_help():
        rprint("The command list:")
        rprint("- 'q' to exit")
        rprint("- 'r' to reset the chat")
        rprint("- 'h' to display these commands")
        rprint()
        rprint("Line editing:")
        rprint("- Use arrow keys to move cursor and edit text")
        rprint("- Up/Down arrows to cycle through history")

    rprint(f"[INFO] Starting chat session with {args.model}.")
    print_help()
    prompt_cache = make_prompt_cache(model, args.max_kv_size)
    while True:
        query = input(">> " if rank == 0 else "")
        if query == "q":
            break
        if query == "r":
            prompt_cache = make_prompt_cache(model, args.max_kv_size)
            continue
        if query == "h":
            print_help()
            continue
        messages = []
        if args.system_prompt is not None:
            messages.append({"role": "system", "content": args.system_prompt})
        messages.append({"role": "user", "content": query})
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        for response in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=args.max_tokens,
            sampler=make_sampler(
                args.temp,
                args.top_p,
                xtc_threshold=args.xtc_threshold,
                xtc_probability=args.xtc_probability,
                xtc_special_tokens=(
                    tokenizer.encode("\n") + list(tokenizer.eos_token_ids)
                ),
            ),
            prompt_cache=prompt_cache,
        ):
            rprint(response.text, flush=True, end="")
        rprint()


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_lm.chat...` directly is deprecated."
        " Use `mlx_lm.chat...` or `python -m mlx_lm chat ...` instead."
    )
    main()
