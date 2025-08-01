import logging
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import re
import hashlib

from ..config import RAGConfig, DocumentMetadata
from ..vectors.embeddings import EmbeddingManager
from ..vectors.vector_store import VectorStore


class DocumentIndexer:
    """
    Indexes documents from the memo directory structure.

    Handles file discovery, content processing, chunking, and embedding generation.
    """

    def __init__(
        self,
        config: RAGConfig,
        embedding_manager: EmbeddingManager,
        vector_store: VectorStore,
    ):
        self.config = config
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store
        self.logger = logging.getLogger(__name__)

        # Track processed files to avoid reprocessing
        self._file_hashes: Dict[str, str] = {}
        self._load_file_hashes()

    def _load_file_hashes(self) -> None:
        """Load file hashes to track changes."""
        hash_file = self.config.cache_dir / "file_hashes.json"
        if hash_file.exists():
            try:
                import json

                with open(hash_file, "r") as f:
                    self._file_hashes = json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load file hashes: {e}")
                self._file_hashes = {}

    def _save_file_hashes(self) -> None:
        """Save file hashes to disk."""
        try:
            import json

            hash_file = self.config.cache_dir / "file_hashes.json"
            with open(hash_file, "w") as f:
                json.dump(self._file_hashes, f)
        except Exception as e:
            self.logger.warning(f"Failed to save file hashes: {e}")

    def _get_file_hash(self, file_path: Path) -> str:
        """Generate hash for file content and metadata."""
        try:
            stat = file_path.stat()
            content_hash = hashlib.md5()

            # Include file size and modification time
            content_hash.update(f"{stat.st_size}:{stat.st_mtime}".encode())

            # Include a sample of content for extra verification
            with open(
                file_path, "r", encoding=self.config.encoding, errors="ignore"
            ) as f:
                sample = f.read(1024)  # First 1KB
                content_hash.update(sample.encode())

            return content_hash.hexdigest()

        except Exception as e:
            self.logger.warning(f"Failed to hash file {file_path}: {e}")
            return str(datetime.now().timestamp())

    async def index_documents(self) -> Dict[str, Any]:
        """
        Index all documents in the memo directory.

        Returns:
            Dictionary with indexing statistics
        """
        self.logger.info("Starting document indexing...")

        stats = {
            "total_files": 0,
            "processed_files": 0,
            "skipped_files": 0,
            "total_chunks": 0,
            "errors": 0,
            "start_time": datetime.now(),
        }

        try:
            # Discover all memo files
            memo_files = self._discover_memo_files()
            stats["total_files"] = len(memo_files)

            self.logger.info(f"Found {len(memo_files)} memo files to process")

            # Process files in batches to control memory usage
            batch_size = self.config.max_concurrent_files

            for i in range(0, len(memo_files), batch_size):
                batch = memo_files[i : i + batch_size]
                batch_stats = await self._process_file_batch(batch)

                # Update stats
                stats["processed_files"] += batch_stats["processed"]
                stats["skipped_files"] += batch_stats["skipped"]
                stats["total_chunks"] += batch_stats["chunks"]
                stats["errors"] += batch_stats["errors"]

                self.logger.info(
                    f"Processed batch {i // batch_size + 1}/{(len(memo_files) - 1) // batch_size + 1}: "
                    f"{batch_stats['processed']} files, {batch_stats['chunks']} chunks"
                )

            # Save file hashes
            self._save_file_hashes()

            stats["end_time"] = datetime.now()
            stats["duration"] = (
                stats["end_time"] - stats["start_time"]
            ).total_seconds()

            self.logger.info(
                f"Indexing completed in {stats['duration']:.2f}s: "
                f"{stats['processed_files']} files, {stats['total_chunks']} chunks"
            )

            return stats

        except Exception as e:
            self.logger.error(f"Indexing failed: {e}")
            stats["error"] = str(e)
            raise

    def _discover_memo_files(self) -> List[Path]:
        """
        Discover all memo files in the directory structure.

        Expected structure: data/memo/YYYY/MM/DD.md
        """
        memo_files = []

        if not self.config.data_root.exists():
            self.logger.warning(
                f"Memo directory does not exist: {self.config.data_root}"
            )
            return memo_files

        # Walk through the directory structure
        for year_dir in self.config.data_root.iterdir():
            if not year_dir.is_dir() or not year_dir.name.isdigit():
                continue

            for month_dir in year_dir.iterdir():
                if not month_dir.is_dir():
                    continue

                for day_file in month_dir.iterdir():
                    if (
                        day_file.is_file()
                        and day_file.suffix in self.config.supported_extensions
                    ):
                        memo_files.append(day_file)

        # Sort by date (newest first)
        memo_files.sort(
            key=lambda p: (p.parent.parent.name, p.parent.name, p.stem), reverse=True
        )

        return memo_files

    async def _process_file_batch(self, files: List[Path]) -> Dict[str, int]:
        """Process a batch of files concurrently."""
        tasks = [self._process_single_file(file_path) for file_path in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        stats = {"processed": 0, "skipped": 0, "chunks": 0, "errors": 0}

        for result in results:
            if isinstance(result, Exception):
                stats["errors"] += 1
                self.logger.error(f"File processing error: {result}")
            elif result:
                if result["processed"]:
                    stats["processed"] += 1
                    stats["chunks"] += result["chunks"]
                else:
                    stats["skipped"] += 1

        return stats

    async def _process_single_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Process a single memo file."""
        try:
            # Check if file needs processing
            current_hash = self._get_file_hash(file_path)
            file_key = str(file_path)

            if (
                file_key in self._file_hashes
                and self._file_hashes[file_key] == current_hash
            ):
                return {"processed": False, "chunks": 0}

            # Read file content
            content = self._read_file_content(file_path)
            if not content.strip():
                self.logger.debug(f"Skipping empty file: {file_path}")
                return {"processed": False, "chunks": 0}

            # Remove existing document from vector store
            await self.vector_store.remove_document(str(file_path))

            # Chunk the content
            chunks = self._chunk_text(content)
            if not chunks:
                return {"processed": False, "chunks": 0}

            # Generate embeddings
            embeddings = self.embedding_manager.embed_texts(chunks)

            # Create metadata for each chunk
            metadatas = []
            for i, chunk in enumerate(chunks):
                metadata = DocumentMetadata.from_file_path(
                    file_path, chunk_index=i, total_chunks=len(chunks)
                )
                metadata.content_preview = (
                    chunk[:200] + "..." if len(chunk) > 200 else chunk
                )
                metadatas.append(metadata)

            # Add to vector store
            await self.vector_store.add_documents(embeddings, chunks, metadatas)

            # Update file hash
            self._file_hashes[file_key] = current_hash

            self.logger.debug(f"Processed {file_path}: {len(chunks)} chunks")
            return {"processed": True, "chunks": len(chunks)}

        except Exception as e:
            self.logger.error(f"Failed to process file {file_path}: {e}")
            raise

    def _read_file_content(self, file_path: Path) -> str:
        """Read and preprocess file content."""
        try:
            with open(
                file_path, "r", encoding=self.config.encoding, errors="ignore"
            ) as f:
                content = f.read()

            # Basic preprocessing
            content = self._preprocess_content(content)
            return content

        except Exception as e:
            self.logger.error(f"Failed to read file {file_path}: {e}")
            return ""

    def _preprocess_content(self, content: str) -> str:
        """Preprocess content before chunking."""
        # Normalize whitespace
        content = re.sub(r"\n\s*\n", "\n\n", content)  # Normalize paragraph breaks
        content = re.sub(r"[ \t]+", " ", content)  # Normalize spaces

        # Remove excessive markdown formatting that might interfere
        content = re.sub(
            r"^#{4,}", "###", content, flags=re.MULTILINE
        )  # Limit header depth

        return content.strip()

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.

        Uses intelligent splitting on sentence boundaries when possible.
        """
        if len(text) <= self.config.chunk_size:
            return [text] if text.strip() else []

        chunks = []

        # Split into sentences first
        sentences = self._split_sentences(text)

        current_chunk = ""
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If single sentence is too long, split it further
            if sentence_length > self.config.chunk_size:
                # Add current chunk if not empty
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                # Split long sentence
                sub_chunks = self._split_long_sentence(sentence)
                chunks.extend(sub_chunks)

                current_chunk = ""
                current_length = 0
                continue

            # Check if adding this sentence would exceed chunk size
            if (
                current_length + sentence_length > self.config.chunk_size
                and current_chunk
            ):
                # Add current chunk
                chunks.append(current_chunk.strip())

                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + sentence
                current_length = len(current_chunk)
            else:
                # Add sentence to current chunk
                current_chunk += sentence
                current_length += sentence_length

        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Filter out chunks that are too small
        chunks = [chunk for chunk in chunks if len(chunk) >= self.config.min_chunk_size]

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - could be improved with proper NLP libraries
        sentences = re.split(r"(?<=[.!?])\s+", text)

        # Handle edge cases and merge very short sentences
        merged_sentences = []
        current = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # If current sentence is very short, try to merge with next
            if len(current) < 50 and current:
                current += " " + sentence
            else:
                if current:
                    merged_sentences.append(current)
                current = sentence

        if current:
            merged_sentences.append(current)

        return merged_sentences

    def _split_long_sentence(self, sentence: str) -> List[str]:
        """Split a long sentence into smaller chunks."""
        chunks = []

        # Try splitting on common punctuation first
        parts = re.split(r"[,;:]", sentence)

        current_chunk = ""
        for part in parts:
            part = part.strip()
            if len(current_chunk + part) > self.config.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = part
            else:
                current_chunk += (", " if current_chunk else "") + part

        if current_chunk:
            chunks.append(current_chunk.strip())

        # If still too long, split by words
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= self.config.chunk_size:
                final_chunks.append(chunk)
            else:
                word_chunks = self._split_by_words(chunk)
                final_chunks.extend(word_chunks)

        return final_chunks

    def _split_by_words(self, text: str) -> List[str]:
        """Split text by words when other methods fail."""
        words = text.split()
        chunks = []
        current_chunk = ""

        for word in words:
            if len(current_chunk + " " + word) > self.config.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = word
            else:
                current_chunk += (" " if current_chunk else "") + word

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _get_overlap_text(self, chunk: str) -> str:
        """Get overlap text from the end of a chunk."""
        if len(chunk) <= self.config.chunk_overlap:
            return chunk + " "

        # Try to get overlap at sentence boundary
        sentences = self._split_sentences(chunk)
        overlap = ""

        for sentence in reversed(sentences):
            if len(overlap + sentence) <= self.config.chunk_overlap:
                overlap = sentence + " " + overlap
            else:
                break

        # If no sentence fits, take last N characters
        if not overlap.strip():
            overlap = chunk[-self.config.chunk_overlap :] + " "

        return overlap

    async def index_single_document(
        self, file_path: Path, force_reindex: bool = False
    ) -> bool:
        """
        Index a single document.

        Args:
            file_path: Path to the document
            force_reindex: Whether to reindex even if up to date

        Returns:
            True if document was indexed, False if skipped
        """
        try:
            # Force reindexing by removing from hash cache
            if force_reindex:
                file_key = str(file_path)
                if file_key in self._file_hashes:
                    del self._file_hashes[file_key]

            result = await self._process_single_file(file_path)

            if result and result["processed"]:
                self._save_file_hashes()
                return True

            return False

        except Exception as e:
            self.logger.error(f"Failed to index single document {file_path}: {e}")
            raise


class TextProcessor:
    """
    Advanced text processing utilities for document indexing.

    Provides methods for content extraction, cleaning, and enhancement.
    """

    @staticmethod
    def extract_metadata_from_content(content: str) -> Dict[str, Any]:
        """Extract metadata from document content."""
        metadata = {}

        # Extract title (first heading or first line)
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if title_match:
            metadata["title"] = title_match.group(1).strip()
        else:
            # Use first non-empty line as title
            lines = content.split("\n")
            for line in lines:
                if line.strip():
                    metadata["title"] = line.strip()[:100]
                    break

        # Extract tags (hashtags or @mentions)
        tags = re.findall(r"#(\w+)", content)
        mentions = re.findall(r"@(\w+)", content)
        metadata["tags"] = list(set(tags))
        metadata["mentions"] = list(set(mentions))

        # Extract dates mentioned in content
        date_patterns = [
            r"\b(\d{4}-\d{2}-\d{2})\b",  # YYYY-MM-DD
            r"\b(\d{1,2}/\d{1,2}/\d{4})\b",  # MM/DD/YYYY
            r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
        ]

        dates = []
        for pattern in date_patterns:
            dates.extend(re.findall(pattern, content, re.IGNORECASE))
        metadata["dates_mentioned"] = list(set(dates))

        # Count words and estimate reading time
        word_count = len(content.split())
        metadata["word_count"] = word_count
        metadata["estimated_reading_time"] = max(1, word_count // 200)  # Assume 200 WPM

        return metadata

    @staticmethod
    def clean_content(content: str) -> str:
        """Clean and normalize content for better processing."""
        # Remove excessive whitespace
        content = re.sub(r"\n\s*\n\s*\n", "\n\n", content)

        # Normalize unicode characters
        content = content.encode("utf-8", errors="ignore").decode("utf-8")

        # Remove or replace special characters that might cause issues
        content = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", content)

        # Normalize quotation marks
        content = re.sub(r'[""' "`]", '"', content)

        # Fix common markdown issues
        content = re.sub(r"\*\*([^*]+)\*\*", r"**\1**", content)  # Fix bold formatting
        content = re.sub(r"\*([^*]+)\*", r"*\1*", content)  # Fix italic formatting

        return content.strip()

    @staticmethod
    def enhance_content_for_search(content: str) -> str:
        """Enhance content to improve search relevance."""
        enhanced = content

        # Expand abbreviations (could be made configurable)
        abbreviations = {
            "w/": "with",
            "w/o": "without",
            "b/c": "because",
            "thru": "through",
            "&": "and",
        }

        for abbrev, expansion in abbreviations.items():
            enhanced = re.sub(
                rf"\b{re.escape(abbrev)}\b", expansion, enhanced, flags=re.IGNORECASE
            )

        # Add context for dates and times
        enhanced = re.sub(
            r"\b(\d{1,2}:\d{2}(?:\s*[AP]M)?)\b",
            r"\1 time",
            enhanced,
            flags=re.IGNORECASE,
        )

        return enhanced


class FileWatcher:
    """
    Optional file system watcher for real-time indexing.

    Monitors the memo directory for changes and triggers reindexing.
    """

    def __init__(self, indexer: DocumentIndexer):
        self.indexer = indexer
        self.logger = logging.getLogger(__name__)
        self._observer = None
        self._running = False

    async def start_watching(self) -> None:
        """Start watching the memo directory for changes."""
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
        except ImportError:
            self.logger.warning(
                "Watchdog not installed. File watching disabled. "
                "Install with: pip install watchdog"
            )
            return

        class MemoEventHandler(FileSystemEventHandler):
            def __init__(self, indexer_ref):
                self.indexer = indexer_ref
                self.logger = logging.getLogger(f"{__name__}.watcher")

            def on_modified(self, event):
                if not event.is_directory:
                    self._handle_file_change(event.src_path)

            def on_created(self, event):
                if not event.is_directory:
                    self._handle_file_change(event.src_path)

            def on_deleted(self, event):
                if not event.is_directory:
                    asyncio.create_task(self._handle_file_deletion(event.src_path))

            def _handle_file_change(self, file_path):
                path = Path(file_path)
                if path.suffix in self.indexer.config.supported_extensions:
                    asyncio.create_task(
                        self.indexer.index_single_document(path, force_reindex=True)
                    )

            async def _handle_file_deletion(self, file_path):
                try:
                    await self.indexer.vector_store.remove_document(file_path)
                    self.logger.info(
                        f"Removed deleted document from index: {file_path}"
                    )
                except Exception as e:
                    self.logger.error(f"Failed to remove deleted document: {e}")

        self._observer = Observer()
        event_handler = MemoEventHandler(self.indexer)

        self._observer.schedule(
            event_handler, str(self.indexer.config.data_root), recursive=True
        )

        self._observer.start()
        self._running = True
        self.logger.info(f"Started watching {self.indexer.config.data_root}")

    def stop_watching(self) -> None:
        """Stop watching for file changes."""
        if self._observer and self._running:
            self._observer.stop()
            self._observer.join()
            self._running = False
            self.logger.info("Stopped file watching")

    def is_running(self) -> bool:
        """Check if file watching is active."""
        return self._running
