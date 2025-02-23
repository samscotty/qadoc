import os
from pathlib import Path

import click

from qadoc.qa import QA


@click.command()
@click.argument("filepath", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--embedding",
    "-e",
    default="text-embedding-ada-002",
    show_default=True,
    help="Name of OpenAI text embedding model to use.",
)
@click.option(
    "--chat",
    "-c",
    default="gpt-3.5-turbo",
    show_default=True,
    help="Name of OpenAI chat model to use.",
)
@click.option(
    "--key",
    "-k",
    prompt="OpenAI API key",
    hide_input=True,
    envvar="OPENAI_API_KEY",
    help="OpenAI API key  [optional: attempts to read from environment]",
)
def main(filepath: Path, embedding: str, chat: str, key: str) -> None:
    """PDF-based Question Answering (Q&A) system with LangChain and OpenAI.

    Given a FILEPATH, attempts to load a single file or recursively loads all files in a directory.

    """
    os.environ.setdefault("OPENAI_API_KEY", key)

    qa = QA.create(embedding, chat)
    pdfs = [filepath] if filepath.is_file() else list(filepath.rglob("*.pdf"))

    with click.progressbar(pdfs, label="Loading documents", show_pos=True) as bar:
        for pdf in bar:
            qa.load(pdf)

    click.echo(
        "Ask questions about your document(s). Type 'quit' or press Ctrl-C to exit."
    )
    while (question := click.prompt(click.style("Q", fg="cyan"), type=str)) != "quit":
        answer = qa.ask(question)
        click.echo(f'{click.style("A", fg="green")}: {answer}')
