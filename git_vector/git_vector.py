import hashlib
import logging
import mimetypes
import os
import pickle
import time
from typing import Any, Dict, List, Tuple

import click
import git
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore [import-untyped]
from tqdm import tqdm

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("codebase_assistant.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def get_repo_hash(repo_dir: str) -> str:
    """Generate a unique hash for the repository directory path."""
    repo_path_bytes = repo_dir.encode("utf-8")
    return hashlib.md5(repo_path_bytes).hexdigest()


def load_embeddings_cache(cache_file: str) -> Dict[str, Any]:
    """Load the embeddings cache from a file or initialize an empty cache."""
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)
        logger.info("Embeddings cache loaded.")
    else:
        cache = {}
        logger.info("Initialized empty embeddings cache.")
    return cache


def save_embeddings_cache(cache: Dict[str, Any], cache_file: str) -> None:
    """Save the embeddings cache to a file."""
    with open(cache_file, "wb") as f:
        pickle.dump(cache, f)
    logger.info("Embeddings cache saved.")


def get_tracked_files(repo_dir: str) -> List[str]:
    """Get a list of all tracked files in the Git repository using GitPython."""
    try:
        repo = git.Repo(repo_dir)
        tracked_files = [
            os.path.join(repo_dir, item.path)  # type: ignore [union-attr]
            for item in repo.tree().traverse()
        ]
        logger.info(f"Found {len(tracked_files)} tracked files using GitPython.")
        return tracked_files
    except git.exc.InvalidGitRepositoryError:
        logger.error("Invalid Git repository.")
        click.echo("The specified directory is not a valid Git repository.")
        return []
    except Exception as e:
        logger.error(f"Error accessing Git repository: {e}")
        click.echo("An error occurred while accessing the Git repository.")
        return []


def is_text_file(file_path: str) -> bool:
    """Check if a file is a text file."""
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type and "text" in mime_type:
        return True
    else:
        # Fallback for files without a MIME type
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                f.read(1024)
            return True
        except Exception:
            return False


def collect_code_files(repo_dir: str) -> List[str]:
    """Collect all text-based code files in the repository."""
    code_files = []
    tracked_files = get_tracked_files(repo_dir)

    if not tracked_files:
        return []

    logger.info("Filtering text files...")

    for file_path in tqdm(tracked_files, desc="Filtering files"):
        if is_text_file(file_path):
            code_files.append(file_path)

    logger.info(f"Collected {len(code_files)} text files for processing.")
    return code_files


def split_into_chunks(text: str, max_tokens: int = 500) -> List[str]:
    """Split text into chunks of approximately max_tokens tokens."""
    tokens = text.split()
    chunks = [
        " ".join(tokens[i : i + max_tokens]) for i in range(0, len(tokens), max_tokens)
    ]
    return chunks


def generate_embeddings(
    text_list: List[str], model_name: str, batch_size: int = 100
) -> List[List[float]]:
    """Generate embeddings for a list of texts using the specified model."""
    embeddings = []
    logger.info(f"Generating embeddings with model: {model_name}")

    for i in tqdm(range(0, len(text_list), batch_size), desc="Embedding chunks"):
        batch = text_list[i : i + batch_size]
        max_retries = 5
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = client.embeddings.create(input=batch, model=model_name)
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
                break  # Exit the retry loop on success
            except RateLimitError:
                retry_count += 1
                wait_time = 2**retry_count  # Exponential backoff
                logger.warning(
                    f"Rate limit exceeded. Retrying in {wait_time} seconds..."
                )
                time.sleep(wait_time)
            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
                break  # Exit the loop on non-rate-limit errors
        else:
            logger.error(f"Max retries exceeded for batch starting at index {i}")
            raise Exception("Max retries exceeded")

    return embeddings


def chunk_and_embed_files(
    file_paths: List[str], cache: Dict[str, Any], cache_file: str, model_name: str
) -> Tuple[List[Dict[str, str]], List[List[float]]]:
    """Chunk the code files and generate embeddings, using cache."""
    code_chunks = []
    embeddings = []
    logger.info("Processing files and generating embeddings...")

    for file_path in tqdm(file_paths, desc="Processing files"):
        try:
            last_modified = os.path.getmtime(file_path)
            cache_entry = cache.get(file_path)

            if cache_entry and cache_entry["last_modified"] == last_modified:
                code_chunks.extend(cache_entry["chunks"])
                embeddings.extend(cache_entry["embeddings"])
                logger.debug(f"Loaded cached embeddings for {file_path}")
            else:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                if not content.strip():
                    continue

                chunks = split_into_chunks(content)
                chunk_embeddings = generate_embeddings(chunks, model_name)

                for chunk, embedding in zip(chunks, chunk_embeddings):
                    code_chunks.append({"content": chunk, "file_path": file_path})
                    embeddings.append(embedding)

                cache[file_path] = {
                    "last_modified": last_modified,
                    "chunks": [
                        {"content": chunk, "file_path": file_path} for chunk in chunks
                    ],
                    "embeddings": chunk_embeddings,
                }
        except Exception as e:
            logger.warning(f"Failed to process {file_path}: {e}")

    save_embeddings_cache(cache, cache_file)

    if not code_chunks:
        logger.warning("No code chunks to process.")

    return code_chunks, embeddings


def get_top_k_similar_chunks(
    query: str,
    code_chunks: List[Dict[str, str]],
    embeddings: List[List[float]],
    embedding_model: str,
    k: int = 5,
) -> List[Dict[str, str]]:
    """Retrieve the top k code chunks most similar to the query."""
    response = client.embeddings.create(input=[query], model=embedding_model)

    query_embedding = response.data[0].embedding
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_k_indices = similarities.argsort()[-k:][::-1]
    top_k_chunks = [code_chunks[i] for i in top_k_indices]
    return top_k_chunks


def construct_prompt(
    question: str, code_contexts: List[Dict[str, str]], max_prompt_tokens: int
) -> str:
    """Construct the prompt for the OpenAI API."""
    context_texts = []

    for context in code_contexts:
        context_text = f"File: {context['file_path']}\nContent:\n{context['content']}\n"
        context_texts.append(context_text)

    context_combined = "\n".join(context_texts)
    prompt = f"""
You are an AI assistant helping a developer understand their codebase.

Code Context:
{context_combined}

Question:
{question}

Answer:
"""

    # Handle token limits
    prompt_tokens_estimate = len(prompt) // 4  # Approximate token count
    if prompt_tokens_estimate > max_prompt_tokens:
        # Truncate context to fit within token limit
        allowed_context_tokens = (
            max_prompt_tokens - len(question) // 4 - 100
        )  # Reserve tokens
        # Truncate context_combined
        context_combined = context_combined[: allowed_context_tokens * 4]
        prompt = f"""
You are an AI assistant helping a developer understand their codebase.

Code Context:
{context_combined}

Question:
{question}

Answer:
"""
    return prompt


def interactive_cli(
    code_chunks: List[Dict[str, str]],
    embeddings: List[List[float]],
    chat_model: str,
    embedding_model: str,
    max_prompt_tokens: int,
    max_response_tokens: int,
) -> None:
    """Start the interactive CLI for the user to chat with the codebase."""
    click.echo(
        "Welcome to the Codebase Assistant. Type your questions below (type 'exit' to quit)."  # noqa: E501
    )
    while True:
        question = click.prompt("\nYour question", type=str)
        if question.lower() == "exit":
            click.echo("Goodbye!")
            break
        logger.info(f"User question: {question}")
        try:
            top_chunks = get_top_k_similar_chunks(
                query=question,
                code_chunks=code_chunks,
                embeddings=embeddings,
                embedding_model=embedding_model,
            )
            prompt = construct_prompt(
                question=question,
                code_contexts=top_chunks,
                max_prompt_tokens=max_prompt_tokens,
            )
            response = None
            max_retries = 5
            retry_count = 0
            while retry_count < max_retries:
                try:
                    response = client.chat.completions.create(
                        model=chat_model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_response_tokens,
                        temperature=0.2,
                    )
                    break  # Exit the retry loop on success
                except RateLimitError:
                    retry_count += 1
                    wait_time = 2**retry_count  # Exponential backoff
                    logger.warning(
                        f"Rate limit exceeded. Retrying in {wait_time} seconds..."
                    )
                    time.sleep(wait_time)
                except Exception as e:
                    logger.error(f"Error during API call: {e}")
                    click.echo("An error occurred while processing your request.")
                    break  # Exit the loop on non-rate-limit errors
            if response:
                answer = response.choices[0].message.content
                click.echo(f"\nAssistant:\n{answer}")
                logger.info("Response provided to the user.")
            else:
                logger.error("Failed to get a response from the API.")
                click.echo("Failed to get a response from the API.")
        except Exception as e:
            logger.error(f"Error processing your question: {e}")
            click.echo("An error occurred while processing your question.")


@click.command()
@click.option(
    "--repo-dir", prompt="Repository path", help="The path to the Git repository."
)
@click.option(
    "--embedding-model",
    default="text-embedding-3-small",
    show_default=True,
    help="The OpenAI model to use for embeddings.",
)
@click.option(
    "--chat-model",
    default="gpt-4o-mini",
    show_default=True,
    help="The OpenAI model to use for chat completions.",
)
@click.option(
    "--max-prompt-tokens",
    default=128_000,
    show_default=True,
    help="Maximum number of tokens for the prompt.",
)
@click.option(
    "--max-response-tokens",
    default=16_384,
    show_default=True,
    help="Maximum number of tokens for the response.",
)
def main(
    repo_dir: str,
    embedding_model: str,
    chat_model: str,
    max_prompt_tokens: int,
    max_response_tokens: int,
) -> None:
    """Main function to run the CLI tool."""
    logger.info("Starting Codebase Assistant...")

    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    cli_dir = os.path.dirname(current_dir)
    cache_dir = os.path.join(cli_dir, "embeddings_cache")
    os.makedirs(cache_dir, exist_ok=True)

    repo_dir = os.path.abspath(repo_dir)
    repo_hash = get_repo_hash(repo_dir)

    cache_file = os.path.join(cache_dir, f"embeddings_cache_{repo_hash}.pkl")
    logger.info(f"Using cache file: {cache_file}")

    if not os.path.isdir(repo_dir):
        logger.error("Invalid repository path.")
        click.echo("The specified repository path is invalid.")
        return

    cache = load_embeddings_cache(cache_file)
    click.echo("Collecting code files...")
    code_files = collect_code_files(repo_dir)

    if not code_files:
        logger.error("No code files found.")
        click.echo(
            "No code files found in the repository or failed to access the repository."
        )
        return

    code_chunks, embeddings = chunk_and_embed_files(
        file_paths=code_files,
        cache=cache,
        cache_file=cache_file,
        model_name=embedding_model,
    )
    if not code_chunks or not embeddings:
        logger.error("Failed to process code files for embeddings.")
        click.echo("Failed to process code files.")
        return

    embeddings_array = np.array(embeddings)
    interactive_cli(
        code_chunks=code_chunks,
        embeddings=embeddings_array,  # type: ignore [arg-type]
        chat_model=chat_model,
        embedding_model=embedding_model,
        max_prompt_tokens=max_prompt_tokens,
        max_response_tokens=max_response_tokens,
    )


if __name__ == "__main__":
    main()
