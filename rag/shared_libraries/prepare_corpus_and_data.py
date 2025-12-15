"""
Corpus management and document preparation for the RAG system.

This script provides a class-based API plus a CLI for:
- creating or retrieving a Vertex AI RAG corpus
- uploading local documents or downloading/uploading documents from URLs
- listing corpus files

CLI flags include:
  --dry-run        : validate configuration and show actions without making changes
  --sample-dir     : path to a directory containing local documents to upload
  --urls-file      : newline-separated file containing URLs to download and upload
  --display-name   : optional display name for corpus
  --no-upload      : create corpus but skip uploads

The script writes `RAG_CORPUS` to the project's `.env` only when setup completes successfully
unless `--dry-run` is used.
"""


import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import requests
from google.auth import default
from google.api_core.exceptions import ResourceExhausted
from dotenv import load_dotenv, set_key
import vertexai
from vertexai.preview import rag


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


load_dotenv()

ENV_FILE_PATH = Path(__file__).parent.parent.parent / ".env"


class CorpusManager:
    EMBEDDING_MODEL = "publishers/google/models/text-embedding-004"
    DEFAULT_CORPUS_NAME = "DocumentCorpus"

    def __init__(self, project_id: str, location: str = "us-central1"):
        self.project_id = project_id
        self.location = location
        self._initialize_vertexai()
        logger.info("Initialized CorpusManager for %s/%s", project_id, location)

    def _initialize_vertexai(self):
        creds, _ = default()
        vertexai.init(project=self.project_id, location=self.location, credentials=creds)

    def get_or_create_corpus(self, display_name: Optional[str] = None):
        name = display_name or self.DEFAULT_CORPUS_NAME
        existing = self._find_corpus_by_name(name)
        if existing:
            logger.info("Found existing corpus: %s", existing.name)
            return existing

        logger.info("Creating new corpus: %s", name)
        embedding_config = rag.EmbeddingModelConfig(publisher_model=self.EMBEDDING_MODEL)
        corpus = rag.create_corpus(display_name=name, description=f"Corpus: {name}", embedding_model_config=embedding_config)
        logger.info("Created corpus: %s", corpus.name)
        return corpus

    def _find_corpus_by_name(self, display_name: str):
        try:
            corpora = rag.list_corpora()
            for c in corpora:
                if c.display_name == display_name:
                    return c
        except (RuntimeError, ValueError) as e:
            logger.warning("Error listing corpora: %s", e)
        return None

    def upload_documents(self, corpus, document_paths: List[str], retry_on_quota: bool = True) -> List[str]:
        uploaded = []
        failed = []
        for path in document_paths:
            if not os.path.exists(path):
                logger.warning("File not found: %s", path)
                failed.append(path)
                continue

            try:
                logger.info("Uploading: %s", path)
                rag.upload_file(corpus_name=corpus.name, path=path)
                uploaded.append(path)
                logger.info("Uploaded: %s", path)
            except ResourceExhausted as e:
                logger.error("Quota exceeded uploading %s: %s", path, e)
                if retry_on_quota:
                    logger.info("Retry is disabled in this script; marking as failed: %s", path)
                failed.append(path)
            except (RuntimeError, ValueError, OSError) as e:
                logger.error("Failed to upload %s: %s", path, e)
                failed.append(path)

        if failed:
            logger.warning("Failed uploads: %s", failed)
        return uploaded

    def list_corpus_files(self, corpus) -> List[str]:
        try:
            files = rag.list_files(corpus_name=corpus.name)
            names = [f.display_name for f in files]
            logger.info("Corpus %s contains %d files", corpus.name, len(names))
            return names
        except (RuntimeError, ValueError, AttributeError) as e:
            logger.error("Error listing files for %s: %s", getattr(corpus, 'name', '<unknown>'), e)
            return []


class DocumentDownloader:
    @staticmethod
    def download_file(url: str, destination: str, timeout: int = 60) -> bool:
        try:
            logger.info("Downloading: %s", url)
            r = requests.get(url, stream=True, timeout=timeout)
            r.raise_for_status()
            with open(destination, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logger.info("Downloaded to %s", destination)
            return True
        except (requests.RequestException, OSError) as e:
            logger.error("Failed to download %s: %s", url, e)
            return False


class CorpusSetup:
    def __init__(self, project_id: str, location: str = "us-central1"):
        self.manager = CorpusManager(project_id, location)
        self.downloader = DocumentDownloader()

    def setup(self, display_name: Optional[str] = None, local_dir: Optional[str] = None, url_list: Optional[List[str]] = None, dry_run: bool = False, upload: bool = True) -> bool:
        corpus = self.manager.get_or_create_corpus(display_name)
        logger.info("Working with corpus: %s", corpus.name)

        files_to_upload: List[str] = []

        if local_dir:
            p = Path(local_dir)
            if not p.exists() or not p.is_dir():
                logger.error("Sample directory not found: %s", local_dir)
                return False
            for f in p.iterdir():
                if f.is_file() and f.suffix.lower() in {".pdf", ".txt", ".md"}:
                    files_to_upload.append(str(f))

        if url_list:
            with tempfile.TemporaryDirectory() as tmpdir:
                for url in url_list:
                    filename = url.split("/")[-1] or "downloaded.pdf"
                    filename = Path(filename).name
                    dest = os.path.join(tmpdir, filename)
                    ok = self.downloader.download_file(url, dest)
                    if ok:
                        files_to_upload.append(dest)

        logger.info("Files to upload: %s", files_to_upload)

        if dry_run:
            logger.info("Dry run enabled; no uploads will be performed")
            return True

        if upload and files_to_upload:
            uploaded = self.manager.upload_documents(corpus, files_to_upload)
            logger.info("Uploaded files: %s", uploaded)

        # final verification
        current_files = self.manager.list_corpus_files(corpus)
        logger.info("Final corpus contains %d files", len(current_files))

        # Persist corpus id to .env
        try:
            set_key(str(ENV_FILE_PATH), "RAG_CORPUS", corpus.name)
            logger.info("Wrote RAG_CORPUS to %s", ENV_FILE_PATH)
        except (OSError, ValueError) as e:
            logger.error("Failed to update .env: %s", e)
            return False

        return True


def load_urls_file(path: str) -> List[str]:
    urls: List[str] = []
    p = Path(path)
    if not p.exists():
        logger.error("URLs file not found: %s", path)
        return urls
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            urls.append(line)
    return urls


def get_configuration() -> tuple:
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    if not project:
        raise ValueError("GOOGLE_CLOUD_PROJECT not set in environment")
    return project, location


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Vertex AI RAG corpus and optionally upload documents.")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration and show actions without performing uploads")
    parser.add_argument("--sample-dir", type=str, help="Local directory with documents to upload")
    parser.add_argument("--urls-file", type=str, help="Path to newline-separated file with document URLs to download and upload")
    parser.add_argument("--display-name", type=str, help="Display name for the RAG corpus")
    parser.add_argument("--no-upload", action="store_true", help="Create corpus but skip uploads")
    return parser


def main(argv: Optional[List[str]] = None):
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    try:
        project, location = get_configuration()
        setup = CorpusSetup(project, location)

        url_list = None
        if args.urls_file:
            url_list = load_urls_file(args.urls_file)

        success = setup.setup(display_name=args.display_name, local_dir=args.sample_dir, url_list=url_list, dry_run=args.dry_run, upload=not args.no_upload)

        if success:
            logger.info("Corpus setup completed successfully")
            print("\n✓ Corpus setup completed successfully")
            print(f"Project: {project}")
            print(f"Location: {location}")
        else:
            logger.error("Corpus setup failed")
            sys.exit(1)

    except ValueError as e:
        logger.error(e)
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()