"""
KNUE Board Ingestor

Fetches recent posts from KNUE web board RSS feeds, parses detail pages,
chunks content, generates embeddings, and upserts into Qdrant with
de-duplication by link.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin

import requests
import structlog

from src.config.config import Config
from src.utils.providers import EmbeddingProvider, ProviderFactory

logger = structlog.get_logger(__name__)


@dataclass
class BoardItem:
    title: str
    link: str
    pub_date: datetime
    description: str = ""


@dataclass
class ParsedDetail:
    title: str
    content: str
    preview_links: List[str]


class KnueBoardIngestor:
    """Pipeline for ingesting KNUE board posts into Qdrant."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.from_env()
        self.logger = logger.bind(component="KnueBoardIngestor")

        # Build embedding service via ProviderFactory
        factory = ProviderFactory()
        self.embedding_service = factory.get_embedding_service(
            self.config.embedding_provider, self.config.get_embedding_service_config()
        )

        # Lazily create Qdrant client on first use
        self._qdrant_client: Optional[Any] = None

    # --------- HTTP helpers ---------
    def _http_get(self, url: str, timeout: int = 15) -> str:

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/123.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        resp = requests.get(url, timeout=timeout, headers=headers)
        resp.raise_for_status()
        # Prefer server-declared encoding; fall back to apparent encoding when missing/latin-1
        if not resp.encoding or resp.encoding.lower() in {"iso-8859-1", "latin-1"}:
            try:
                apparent = resp.apparent_encoding  # type: ignore[attr-defined]
            except Exception:
                apparent = None
            if apparent:
                resp.encoding = apparent
        return resp.text

    # --------- RSS parsing ---------
    def _parse_rss(self, xml_text: str) -> List[BoardItem]:
        """Parse RSS/Atom-like XML robustly, handling namespaces and variants.

        Extracts title, link, pubDate (or dc:date/updated), and a plain-text
        description (HTML stripped).
        """
        import xml.etree.ElementTree as ET

        def local(tag: str) -> str:
            return tag.split("}", 1)[-1] if "}" in tag else tag

        def first_child_text(elem: ET.Element, candidates: List[str]) -> Optional[str]:
            for child in list(elem):
                name = local(child.tag).lower()
                if name in candidates:
                    text = (child.text or "").strip()
                    if text:
                        return text
            return None

        items: List[BoardItem] = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            # As a last resort, try to strip control chars and parse again
            cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", xml_text)
            root = ET.fromstring(cleaned)

        # Collect <item> or <entry>
        raw_items: List[ET.Element] = []
        for el in root.iter():
            lname = local(el.tag).lower()
            if lname in {"item", "entry"}:
                raw_items.append(el)

        for it in raw_items:
            title = first_child_text(it, ["title"]) or ""
            # link: prefer <link> text; in Atom link is attribute href
            link = ""
            for child in list(it):
                if local(child.tag).lower() == "link":
                    href = (child.text or "").strip()
                    if not href:
                        href = child.attrib.get("href", "").strip()
                    if href:
                        link = href
                        break
            if not link:
                # try guid
                guid = first_child_text(it, ["guid", "id"]) or ""
                if guid:
                    link = guid

            # date: pubDate (RSS) or dc:date/updated (Atom/DC)
            date_raw = (
                first_child_text(it, ["pubdate"])
                or first_child_text(it, ["date"])
                or first_child_text(it, ["updated"])
                or ""
            )

            # description: ensure plain text
            desc_raw = first_child_text(it, ["description", "summary"]) or ""
            description = self._strip_html(desc_raw).strip()

            # Skip if required fields missing
            if not (title and link and date_raw):
                continue

            # Parse date robustly
            dt: Optional[datetime] = None
            try:
                # Try RFC822 first
                dt = parsedate_to_datetime(date_raw)
            except Exception:
                dt = None
            if dt is None:
                # Try ISO8601
                iso = date_raw.replace("Z", "+00:00")
                try:
                    dt = datetime.fromisoformat(iso)
                except Exception:
                    dt = None
            if dt is None:
                # Give up on this item
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)

            items.append(
                BoardItem(
                    title=self._strip_html(title).strip(),
                    link=self._strip_html(link).strip(),
                    pub_date=dt,
                    description=description,
                )
            )

        return items

    # --------- HTML parsing ---------
    def _strip_html(self, html: str) -> str:
        """Convert HTML (possibly entity-escaped) to readable plain text.

        - Decodes HTML entities (preserving user-intended bracketed tokens like
          "&lt;행정예고&gt;" as plain text "행정예고").
        - Drops all tags/CSS/scripts.
        - Normalizes lists to simple "- " bullets and adds line breaks for
          block elements.
        - Collapses excessive whitespace while preserving paragraph breaks.
        """
        from html import unescape
        from html.parser import HTMLParser

        if not html:
            return ""

        # Protect textual angle-bracket tokens like &lt;공지&gt; so they don't get
        # mistaken for tags after unescape. Replace with placeholders.
        protected = re.sub(r"&lt;([^\s<>/=]+?)&gt;", r"[TEXT:\1]", html)

        # Decode entities (once is usually enough; double decode can eat placeholders)
        text = unescape(protected)

        # Parse HTML and ignore script/style content via the parser

        class _Extractor(HTMLParser):
            def __init__(self):
                super().__init__(convert_charrefs=True)
                self.parts: List[str] = []
                self._skip_depth: int = 0  # inside <script>/<style>

            def handle_starttag(self, tag, attrs):
                t = tag.lower()
                if t in {"script", "style"}:
                    # Enter skip mode for script/style content
                    self._skip_depth += 1
                    return
                if self._skip_depth > 0:
                    return
                if t in {"br"}:
                    self.parts.append("\n")
                elif t in {
                    "p",
                    "div",
                    "tr",
                    "table",
                    "thead",
                    "tbody",
                    "h1",
                    "h2",
                    "h3",
                    "h4",
                    "h5",
                    "h6",
                    "blockquote",
                }:
                    self.parts.append("\n")
                elif t == "li":
                    self.parts.append("\n- ")

            def handle_endtag(self, tag):
                t = tag.lower()
                if t in {"script", "style"}:
                    if self._skip_depth > 0:
                        self._skip_depth -= 1
                    return
                if self._skip_depth > 0:
                    return
                if t in {"p", "div", "tr", "ul", "ol", "table"}:
                    self.parts.append("\n")

            def handle_data(self, data):
                if self._skip_depth > 0:
                    return
                if data:
                    self.parts.append(data)

        # Normalize malformed end tags like '</script foo="bar">' or '</script >'
        # so that HTMLParser can correctly recognize them.
        text = re.sub(r"(?is)</(script|style)\b[^>]*>", r"</\1>", text)

        parser = _Extractor()
        try:
            parser.feed(text)
            parser.close()
            text = "".join(parser.parts)
        except Exception:
            # Fallback: strip tags with regex if parser fails
            text = re.sub(r"(?is)<[^>]+>", " ", text)

        # Restore protected textual tokens without angle brackets
        text = text.replace("[TEXT:", "[[TEXT:")  # guard during second unescape
        text = text.replace("]", "]]", 1) if text.startswith("[TEXT:") else text
        text = text.replace("[[TEXT:", "[TEXT:")
        text = re.sub(r"\[TEXT:([^\]]+)\]", r"\1", text)

        # Normalize whitespace
        text = text.replace("\r", "\n")
        # Trim spaces around newlines
        text = re.sub(r"[ \t]+\n", "\n", text)
        # Collapse multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Collapse internal spaces
        text = re.sub(r"[ \t]{2,}", " ", text)
        # Map common icon bullets to '- '
        text = re.sub(r"[•▪◦·●○◆◇■□▲△▶▷►▸▹▻❖❥❯❱➤➔➜➣➢➤➧➨➩➪➲➣➤]", " - ", text)
        # Remove residual tag-name artifacts (e.g., 'li', 'tbody', 'tr', 'strong', possibly concatenated)
        tag_tokens = (
            "tbody",
            "thead",
            "table",
            "tr",
            "th",
            "td",
            "li",
            "ul",
            "ol",
            "strong",
            "em",
            "span",
            "div",
            "p",
            "br",
            "hr",
        )
        tag_pattern = re.compile(
            r"(?i)(?<![A-Za-z])(?:" + "|".join(tag_tokens) + r")+(?![A-Za-z])"
        )
        text = tag_pattern.sub(" ", text)
        # Normalize whitespace again after removals
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s+", "\n", text)
        return text.strip()

    def _parse_detail(self, html: str, base_url: str) -> ParsedDetail:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")

        # title: .p-table__subject_text
        title = ""
        title_elem = soup.select_one(".p-table__subject_text")
        if title_elem:
            title = self._strip_html(str(title_elem)).strip()

        # content: .p-table__content (keep text only)
        content = ""
        content_elem = soup.select_one(".p-table__content")
        if content_elem:
            content = self._strip_html(str(content_elem)).strip()

        # attachments: <a class="p-attach__preview" href="...">
        preview_links: List[str] = []
        attach_links = soup.select("a.p-attach__preview[href]")
        for link in attach_links:
            href = link.get("href")
            if href:
                preview_links.append(urljoin(base_url, href))

        return ParsedDetail(title=title, content=content, preview_links=preview_links)

    # --------- Skip rules ---------
    def _should_skip_item(
        self, board_idx: int, title: str
    ) -> Tuple[bool, Optional[str]]:
        prefixes = getattr(self.config, "board_skip_prefix_map", {}).get(board_idx, ())
        for pref in prefixes:
            if title.startswith(pref):
                return True, pref
        return False, None

    # --------- Chunking ---------
    def _chunk_text(self, text: str, size: int, overlap: int) -> List[str]:
        if size <= 0:
            return [text]
        if not text:
            return []
        chunks: List[str] = []
        start = 0
        n = len(text)
        while start < n:
            end = min(n, start + size)
            chunks.append(text[start:end])
            if end == n:
                break
            start = max(0, end - overlap)
        return chunks

    # --------- Qdrant client helpers ---------
    @property
    def qdrant_client(self) -> Any:  # lazy
        from qdrant_client import QdrantClient

        if self._qdrant_client is not None:
            return self._qdrant_client

        self._qdrant_client = QdrantClient(
            url=self.config.qdrant_cloud_url, api_key=self.config.qdrant_api_key
        )

        return self._qdrant_client

    def _ensure_board_collection(self, vector_size: int) -> None:
        from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

        name = self.config.qdrant_board_collection
        if not self.qdrant_client.collection_exists(name):
            self.logger.info(
                "Creating Qdrant collection for boards",
                collection=name,
                vector_size=vector_size,
            )
            self.qdrant_client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

        # Ensure required payload indexes exist for filtering
        indexes = {
            "link": PayloadSchemaType.KEYWORD,
            "source": PayloadSchemaType.KEYWORD,
            "board_idx": PayloadSchemaType.INTEGER,
            # Store pubDate as timestamp but index as FLOAT for range filters
            "pubDate": PayloadSchemaType.FLOAT,
        }
        for field, schema in indexes.items():
            try:
                self.qdrant_client.create_payload_index(
                    collection_name=name, field_name=field, field_schema=schema
                )
                self.logger.info(
                    "Created payload index",
                    collection=name,
                    field=field,
                    schema=str(schema),
                )
            except Exception as e:
                # If pubDate index creation fails, try to recreate it
                if field == "pubDate":
                    self.logger.info(
                        "Recreating pubDate index for FLOAT compatibility",
                        collection=name,
                        field=field,
                    )
                    try:
                        # Try to delete existing index first
                        self.qdrant_client.delete_payload_index(
                            collection_name=name, field_name=field
                        )
                    except Exception:
                        pass  # Index might not exist

                    # Create new index with FLOAT type
                    self.qdrant_client.create_payload_index(
                        collection_name=name,
                        field_name=field,
                        field_schema=PayloadSchemaType.FLOAT,
                    )
                    self.logger.info(
                        "Recreated payload index",
                        collection=name,
                        field=field,
                        schema="FLOAT",
                    )
                else:
                    # Likely already exists; log at debug level
                    self.logger.debug(
                        "Payload index ensure",
                        collection=name,
                        field=field,
                        status="exists_or_failed",
                        error=str(e),
                    )

    def _delete_old_items(self, board_idx: int, cutoff: datetime) -> int:
        """Delete points older than cutoff for a specific board.

        Uses server-side filtering for efficiency. Returns count
        of deleted points.
        """
        from qdrant_client.models import (
            FieldCondition,
            Filter,
            MatchValue,
            PointIdsList,
            Range,
        )

        name = self.config.qdrant_board_collection

        try:
            scroll_filter = Filter(
                must=[
                    FieldCondition(key="source", match=MatchValue(value="knue_board")),
                    FieldCondition(key="board_idx", match=MatchValue(value=board_idx)),
                    FieldCondition(key="pubDate", range=Range(lt=cutoff.timestamp())),
                ]
            )
        except Exception as e:
            if "Index required but not found" in str(e):
                self.logger.info(
                    "Recreating pubDate index for range queries",
                    board_idx=board_idx,
                    collection=name,
                )
                try:
                    # Delete existing index
                    self.qdrant_client.delete_payload_index(
                        collection_name=name, field_name="pubDate"
                    )
                except Exception:
                    pass

                # Create new FLOAT index
                from qdrant_client.models import PayloadSchemaType

                self.qdrant_client.create_payload_index(
                    collection_name=name,
                    field_name="pubDate",
                    field_schema=PayloadSchemaType.FLOAT,
                )

                # Retry filter creation
                scroll_filter = Filter(
                    must=[
                        FieldCondition(
                            key="source", match=MatchValue(value="knue_board")
                        ),
                        FieldCondition(
                            key="board_idx", match=MatchValue(value=board_idx)
                        ),
                        FieldCondition(
                            key="pubDate", range=Range(lt=cutoff.timestamp())
                        ),
                    ]
                )
            else:
                raise

        # To get an accurate count, we scroll to find all matching point IDs first.
        ids_to_delete: List[str] = []
        offset = None
        while True:
            try:
                points, offset = self.qdrant_client.scroll(
                    collection_name=name,
                    scroll_filter=scroll_filter,
                    limit=1000,
                    with_payload=False,
                    with_vectors=False,
                    offset=offset,
                )
            except Exception as e:
                if "Index required but not found" in str(e):
                    self.logger.info(
                        "Recreating pubDate index for range queries during scroll",
                        board_idx=board_idx,
                        collection=name,
                    )
                    try:
                        # Delete existing index
                        self.qdrant_client.delete_payload_index(
                            collection_name=name, field_name="pubDate"
                        )
                    except Exception:
                        pass

                    # Create new FLOAT index
                    from qdrant_client.models import PayloadSchemaType

                    self.qdrant_client.create_payload_index(
                        collection_name=name,
                        field_name="pubDate",
                        field_schema=PayloadSchemaType.FLOAT,
                    )

                    # Retry scroll
                    points, offset = self.qdrant_client.scroll(
                        collection_name=name,
                        scroll_filter=scroll_filter,
                        limit=1000,
                        with_payload=False,
                        with_vectors=False,
                        offset=offset,
                    )
                else:
                    raise
            if not points:
                break

            ids_to_delete.extend([str(p.id) for p in points])

            if offset is None:
                break

        if not ids_to_delete:
            return 0

        self.qdrant_client.delete(
            collection_name=name,
            points_selector=PointIdsList(points=ids_to_delete),
        )
        return len(ids_to_delete)

    def _has_points_for_board(self, board_idx: int) -> bool:
        """Return True if the collection already has any point for this board index."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        name = self.config.qdrant_board_collection
        try:
            points, _ = self.qdrant_client.scroll(
                collection_name=name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="source", match=MatchValue(value="knue_board")
                        ),
                        FieldCondition(
                            key="board_idx", match=MatchValue(value=board_idx)
                        ),
                    ]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            return bool(points)
        except Exception:
            # If scroll fails (e.g., collection just created), treat as empty
            return False

    def _delete_by_link(self, link: str) -> int:
        # Scroll and delete all points with payload.link == link
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        name = self.config.qdrant_board_collection
        # Collect IDs
        ids: List[str] = []
        offset = None
        while True:
            points, offset = self.qdrant_client.scroll(
                collection_name=name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="link", match=MatchValue(value=link))]
                ),
                limit=200,
                with_payload=False,
                with_vectors=False,
                offset=offset,
            )
            if not points:
                break
            ids.extend([p.id for p in points])
            if offset is None:
                break
        if not ids:
            return 0

        self.qdrant_client.delete(collection_name=name, points_selector=ids)
        return len(ids)

    # --------- Ingest main ---------
    def ingest(
        self,
        board_indices: Optional[Iterable[int]] = None,
        *,
        force_full: bool = False,
        min_pubdate: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        idxs = (
            tuple(board_indices)
            if board_indices is not None
            else self.config.board_indices
        )

        # Choose embedding dimension from provider
        model_info = self.embedding_service.get_model_info()
        vector_dim = int(model_info.get("dimension", self.config.vector_size))
        self._ensure_board_collection(vector_dim)

        total_processed = 0
        total_upserted = 0
        total_deleted = 0
        failures: List[str] = []

        now = datetime.now(timezone.utc)
        age_limit = now - timedelta(days=self.config.board_max_age_days)

        for board_idx in idxs:
            rss_url = self.config.board_rss_template.format(board_idx=board_idx)
            try:
                rss_xml = self._http_get(rss_url)
            except Exception as e:
                self.logger.warning(
                    "Failed to fetch RSS",
                    board_idx=board_idx,
                    url=rss_url,
                    error=str(e),
                )
                continue

            items = self._parse_rss(rss_xml)
            self.logger.info("Parsed RSS items", board_idx=board_idx, items=len(items))

            # Purge items older than configured retention for this board
            # Only attempt deletion if there are existing points for this board
            if self._has_points_for_board(board_idx):
                retention_days = getattr(self.config, "board_retention_days", 730)
                cutoff_dt = now - timedelta(days=retention_days)
                try:
                    deleted_old = self._delete_old_items(board_idx, cutoff_dt)
                    if deleted_old:
                        self.logger.info(
                            "Deleted old items",
                            board_idx=board_idx,
                            older_than_days=retention_days,
                            deleted=deleted_old,
                        )
                    total_deleted += deleted_old
                except Exception as e:
                    self.logger.warning(
                        "Failed to delete old items", board_idx=board_idx, error=str(e)
                    )
            # Determine indexing mode and filter
            first_run = not self._has_points_for_board(board_idx)
            if min_pubdate is not None:
                recent_items = [it for it in items if it.pub_date >= min_pubdate]
                self.logger.info(
                    "Applying min_pubdate filter",
                    board_idx=board_idx,
                    min_pubdate=min_pubdate.isoformat(),
                    items_after=len(recent_items),
                )
            elif force_full or first_run:
                self.logger.info(
                    "First run for board; indexing full RSS",
                    board_idx=board_idx,
                    items=len(items),
                )
                recent_items = items
            else:
                # Filter by age for subsequent runs
                recent_items = [it for it in items if it.pub_date >= age_limit]

            # Batch accumulators across items for efficient embedding (adaptive)
            batch_texts: List[str] = []
            batch_pending: List[Tuple[str, Dict[str, Any]]] = []  # (point_id, payload)
            max_batch = max(1, getattr(self.config, "board_embed_batch_size", 32))
            dynamic_batch = max_batch
            retry_max = max(0, int(getattr(self.config, "board_embed_retry_max", 3)))
            backoff_base = float(getattr(self.config, "board_embed_backoff_base", 0.5))

            def flush_batch():
                nonlocal batch_texts, batch_pending, total_upserted, dynamic_batch
                if not batch_texts:
                    return
                start = 0
                while start < len(batch_texts):
                    end = min(len(batch_texts), start + dynamic_batch)
                    sub_texts = batch_texts[start:end]
                    sub_pending = batch_pending[start:end]

                    attempt = 0
                    while True:
                        try:
                            vectors = self.embedding_service.generate_embeddings_batch(
                                sub_texts
                            )
                            points = []
                            for (point_id, payload), vec in zip(sub_pending, vectors):
                                points.append(
                                    {"id": point_id, "vector": vec, "payload": payload}
                                )
                            self.qdrant_client.upsert(
                                collection_name=self.config.qdrant_board_collection,
                                points=points,
                            )
                            total_upserted += len(points)
                            # On success, gently increase toward max
                            if dynamic_batch < max_batch:
                                dynamic_batch = min(max_batch, dynamic_batch + 1)
                            break
                        except Exception as e:
                            # Reduce batch and retry with simple exponential backoff
                            if dynamic_batch > 1:
                                dynamic_batch = max(1, dynamic_batch // 2)
                            if attempt >= retry_max:
                                raise
                            delay = backoff_base * (2**attempt)
                            self.logger.warning(
                                "Embedding sub-batch failed; backing off",
                                error=str(e),
                                attempt=attempt + 1,
                                next_delay_s=delay,
                                new_batch_size=dynamic_batch,
                            )
                            import time as _t

                            _t.sleep(delay)
                            attempt += 1

                    start = end

                # reset accumulators
                batch_texts = []
                batch_pending = []

            for item in recent_items:
                total_processed += 1
                try:
                    html = self._http_get(item.link)
                    detail = self._parse_detail(html, item.link)

                    # If RSS title empty, fallback
                    title = detail.title or item.title
                    # Use RSS description as primary content (plain text)
                    content = item.description
                    if not content:
                        self.logger.info(
                            "Empty content; skipping",
                            link=item.link,
                            board_idx=board_idx,
                        )
                        continue

                    safe_title = (title or "").strip()
                    should_skip, matched = self._should_skip_item(board_idx, safe_title)
                    if should_skip:
                        self.logger.info(
                            "Skipping item by board-specific prefix rule",
                            board_idx=board_idx,
                            matched_prefix=matched,
                            title=safe_title,
                            link=item.link,
                        )
                        continue

                    # De-dup by link
                    deleted = self._delete_by_link(item.link)
                    total_deleted += deleted

                    # Chunk content using global chunk settings
                    chunks = self._chunk_text(
                        content, self.config.chunk_threshold, self.config.chunk_overlap
                    )

                    # Queue chunks for batch embedding and upsert
                    base_id = str(uuid.uuid5(uuid.NAMESPACE_URL, item.link))
                    total_chunks = len(chunks)
                    for idx, chunk in enumerate(chunks):
                        point_id = str(
                            uuid.uuid5(uuid.NAMESPACE_DNS, f"{base_id}-{idx}")
                        )
                        payload = {
                            "title": title,
                            "content": chunk,
                            "link": item.link,
                            "pubDate": item.pub_date.timestamp(),
                            "preview_link": detail.preview_links,
                            "board_idx": board_idx,
                            "chunk_index": idx,
                            "total_chunks": total_chunks,
                            "source": "knue_board",
                        }
                        batch_texts.append(chunk)
                        batch_pending.append((point_id, payload))
                        if len(batch_texts) >= dynamic_batch:
                            flush_batch()

                except Exception as e:
                    failures.append(item.link)
                    self.logger.warning(
                        "Failed to process item", link=item.link, error=str(e)
                    )

            # Flush any remaining queued chunks
            flush_batch()

        return {
            "board_indices": list(idxs),
            "processed": total_processed,
            "deleted": total_deleted,
            "upserted": total_upserted,
            "failed": failures,
        }

    def _purge_board_indices(self, board_indices: Iterable[int]) -> int:
        """Delete all points for the given board indices.

        Returns the number of deleted points.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        name = self.config.qdrant_board_collection
        total = 0
        for board_idx in board_indices:
            ids: List[str] = []
            offset = None
            while True:
                points, offset = self.qdrant_client.scroll(
                    collection_name=name,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="source", match=MatchValue(value="knue_board")
                            ),
                            FieldCondition(
                                key="board_idx", match=MatchValue(value=board_idx)
                            ),
                        ]
                    ),
                    limit=1000,
                    with_payload=False,
                    with_vectors=False,
                    offset=offset,
                )
                if not points:
                    break
                ids.extend([p.id for p in points])
                if offset is None:
                    break
            if ids:
                self.qdrant_client.delete(collection_name=name, points_selector=ids)
                total += len(ids)
        return total

    def reindex(
        self,
        board_indices: Optional[Iterable[int]] = None,
        *,
        drop_collection: bool = False,
    ) -> Dict[str, Any]:
        """Reindex boards into the board collection.

        - If drop_collection is True and no specific board indices are provided,
          the entire board collection is deleted and recreated (handles vector size changes).
        - If specific board indices are provided and drop_collection is False,
          only those boards' points are purged before full ingest.
        - Always performs a full ingest (no age filtering).
        """
        idxs = (
            tuple(board_indices)
            if board_indices is not None
            else self.config.board_indices
        )

        # Ensure embedding service and compute dimension
        model_info = self.embedding_service.get_model_info()
        vector_dim = int(model_info.get("dimension", self.config.vector_size))

        name = self.config.qdrant_board_collection
        if drop_collection and (
            board_indices is None or len(idxs) == len(self.config.board_indices)
        ):
            try:
                if self.qdrant_client.collection_exists(name):
                    self.logger.info("Deleting board collection", collection=name)
                    self.qdrant_client.delete_collection(collection_name=name)
            except Exception as e:
                self.logger.warning(
                    "Failed to delete collection", collection=name, error=str(e)
                )

        # Ensure collection and indexes exist (recreated if dropped)
        self._ensure_board_collection(vector_dim)

        # If not dropping entire collection but reindexing subset, purge those boards
        if not drop_collection and idxs:
            deleted = self._purge_board_indices(idxs)
            if deleted:
                self.logger.info(
                    "Purged existing points for boards",
                    deleted=deleted,
                    boards=list(idxs),
                )

        # Perform full ingest for the specified boards, but exclude older than retention
        now = datetime.now(timezone.utc)
        retention_days = getattr(self.config, "board_retention_days", 730)
        cutoff_dt = now - timedelta(days=retention_days)
        return self.ingest(idxs, force_full=True, min_pubdate=cutoff_dt)
