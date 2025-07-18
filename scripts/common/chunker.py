"""テキスト分割モジュール"""
import re
import tiktoken
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class Chunk:
    """チャンクデータクラス"""
    content: str
    metadata: Dict[str, any]
    token_count: int


class MarkdownChunker:
    """Markdown対応のチャンカー"""

    def __init__(self, max_chunk_size: int, chunk_overlap: int = 200, min_chunk_size: int = 100):
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        # 短いセクションを結合するための閾値（最大チャンクサイズの30%）
        self.merge_threshold = int(max_chunk_size * 0.3)

    def count_tokens(self, text: str) -> int:
        """トークン数をカウント"""
        return len(self.tokenizer.encode(text))

    def split_by_headings(self, content: str) -> List[Tuple[str, Dict]]:
        """見出しでテキストを分割"""
        heading_pattern = r'^(#{1,6})\s+(.+)$'

        sections = []
        current_section = ""
        current_heading = ""
        current_level = 0

        lines = content.split('\n')

        for line in lines:
            match = re.match(heading_pattern, line, re.MULTILINE)
            if match:
                if current_section.strip():
                    sections.append((current_section.strip(), {
                        'heading': current_heading,
                        'heading_level': current_level
                    }))

                current_level = len(match.group(1))
                current_heading = match.group(2).strip()
                current_section = line + '\n'
            else:
                current_section += line + '\n'

        if current_section.strip():
            sections.append((current_section.strip(), {
                'heading': current_heading,
                'heading_level': current_level
            }))

        return sections

    def chunk_text(self, text: str, file_metadata: Dict) -> List[Chunk]:
        """テキストをチャンクに分割"""
        chunks = []

        sections = self.split_by_headings(text)

        if not sections:
            # 見出しがない場合は全体を1つのセクションとして扱う
            sections = [(text, {'heading': '', 'heading_level': 0})]

        merged_sections = self._merge_small_sections(sections)

        for section_text, section_metadata in merged_sections:
            token_count = self.count_tokens(section_text)

            if token_count <= self.max_chunk_size:
                chunks.append(Chunk(
                    content=section_text,
                    metadata={
                        **file_metadata,
                        **section_metadata,
                        'chunk_index': len(chunks),
                        'is_partial': False
                    },
                    token_count=token_count
                ))
            else:
                sub_chunks = self._split_large_section(
                    section_text, section_metadata, file_metadata, len(chunks)
                )
                chunks.extend(sub_chunks)

        return chunks

    def _split_large_section(self, text: str, section_metadata: Dict, file_metadata: Dict, start_index: int) -> List[Chunk]:
        """大きなセクションを分割"""
        chunks = []

        paragraphs = text.split('\n\n')

        current_chunk = ""
        current_tokens = 0

        for paragraph in paragraphs:
            paragraph_tokens = self.count_tokens(paragraph)

            if current_tokens + paragraph_tokens > self.max_chunk_size and current_chunk:
                chunks.append(Chunk(
                    content=current_chunk.strip(),
                    metadata={
                        **file_metadata,
                        **section_metadata,
                        'chunk_index': start_index + len(chunks),
                        'is_partial': True
                    },
                    token_count=current_tokens
                ))

                # オーバーラップを考慮した次のチャンク開始
                if self.chunk_overlap > 0 and current_tokens > self.chunk_overlap:
                    overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                    current_chunk = overlap_text + '\n\n' + paragraph
                    current_tokens = self.count_tokens(current_chunk)
                else:
                    current_chunk = paragraph
                    current_tokens = paragraph_tokens
            else:
                if current_chunk:
                    current_chunk += '\n\n' + paragraph
                else:
                    current_chunk = paragraph
                current_tokens += paragraph_tokens

        if current_chunk and current_tokens >= self.min_chunk_size:
            chunks.append(Chunk(
                content=current_chunk.strip(),
                metadata={
                    **file_metadata,
                    **section_metadata,
                    'chunk_index': start_index + len(chunks),
                    'is_partial': True
                },
                token_count=current_tokens
            ))

        return chunks

    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """オーバーラップ用のテキストを取得"""
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= overlap_tokens:
            return text

        overlap_token_ids = tokens[-overlap_tokens:]
        return self.tokenizer.decode(overlap_token_ids)

    def _merge_small_sections(self, sections: List[Tuple[str, Dict]]) -> List[Tuple[str, Dict]]:
        """短いセクションを結合"""
        if not sections:
            return sections

        merged = []
        current_content = ""
        current_metadata = dict()
        current_tokens = 0

        for section_text, section_metadata in sections:
            section_tokens = self.count_tokens(section_text)

            if not current_content:
                current_content = section_text
                current_metadata = section_metadata.copy()
                current_tokens = section_tokens
                continue

            combined_tokens = current_tokens + section_tokens

            # 結合条件：
            # 1. 両方が短いセクション（閾値以下）
            # 2. 結合しても最大サイズを超えない
            # 3. 見出しレベルの差が2以内（階層が近い）
            level_diff = abs(section_metadata['heading_level'] - current_metadata['heading_level'])
            should_merge = (
                current_tokens < self.merge_threshold and
                section_tokens < self.merge_threshold and
                combined_tokens <= self.max_chunk_size and
                level_diff <= 2
            )

            if should_merge:
                current_content += '\n\n' + section_text
                current_tokens = combined_tokens
                # 複数の見出しを含む場合のメタデータ更新
                if 'merged_headings' not in current_metadata:
                    current_metadata['merged_headings'] = current_metadata['heading']
                current_metadata['merged_headings'] = current_metadata['merged_headings'] + ', ' + section_metadata['heading']
                # 最上位レベルの見出しを保持
                if section_metadata['heading_level'] < current_metadata['heading_level']:
                    current_metadata['heading'] = section_metadata['heading']
                    current_metadata['heading_level'] = section_metadata['heading_level']
            else:
                merged.append((current_content, current_metadata))
                current_content = section_text
                current_metadata = section_metadata.copy()
                current_tokens = section_tokens

        if current_content:
            merged.append((current_content, current_metadata))

        return merged
