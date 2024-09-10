# Git-Vector

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Git-Vector** is a command-line tool that allows developers to interact with their Git repositories using OpenAI models. It provides a conversational interface to help you understand and navigate your codebase by leveraging the full context of your repository.

## Table of Contents

- [Git-Vector](#git-vector)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Configuration](#configuration)
  - [License](#license)
  - [Author](#author)
  - [Acknowledgments](#acknowledgments)

## Features

- **Conversational Interface**: Interactively ask questions about your codebase and receive detailed explanations.
- **Full Repository Context**: The tool indexes your entire Git repository to provide context-aware responses.
- **Caching Mechanism**: Embeddings are cached to improve performance on subsequent runs.
- **Customizable Models**: Supports different OpenAI models for both embeddings and chat completions.
- **Configurable Parameters**: Adjust maximum tokens for prompts and responses to suit your needs.

## Installation

```bash
pip install git-vector
```

## Usage

1. **Set Up Environment Variables**:

   Export your OpenAI API key as an environment variable (or use a `.env` file):

   ```
   export OPENAI_API_KEY=your_api_key_here
   ```

2. **Run the Application**:

   ```
   git-vector --repo-dir /path/to/your/git/repository
   ```

   This will start the intera`tive CLI for the user to chat with the codebase.

## Configuration

- `--repo-dir`: (Required) The path to the Git repository.
- `--embedding-model`: The OpenAI model to use for embeddings (default: `text-embedding-3-small`).
- `--chat-model`: The OpenAI model to use for chat completions (default: `gpt-4o-mini`).
- `--max-prompt-tokens`: Maximum number of tokens for the prompt (default: `2000`).
- `--max-response-tokens`: Maximum number of tokens for the response (default: `500`).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author

- Markus Blomqvist - [blomqvist_markus@hotmail.com](mailto:blomqvist_markus@hotmail.com)

## Acknowledgments

- [OpenAI](https://openai.com) for providing the API that powers this tool.
- [GitPython](https://gitpython.readthedocs.io/en/stable/) for interacting with Git repositories.
- [NumPy](https://numpy.org/) and [scikit-learn](https://scikit-learn.org/) for numerical and machine learning
  2024-09-25 00:12:15,639 [INFO] Response provided to the user.
