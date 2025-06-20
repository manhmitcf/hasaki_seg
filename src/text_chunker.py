
import re
from typing import List
from dataclasses import dataclass
from enum import Enum

class ContentType(Enum):
    """Enum for different types of content chunks"""
    DESCRIPTION = "description"
    GENERAL_INFO = "general_info"
    COMMENT = "comment"
    SPECIFICATION = "specification"
    INGREDIENT = "ingredient"
    GUIDE = "guide"


@dataclass
class ChunkData:
    """Data class for chunk information"""
    text: str
    chunk_id: int
    content_type: ContentType

class TextChunker:
    """
    Utility class for chunking long text into smaller, semantically meaningful pieces
    """
    
    def __init__(self, chunk_size: int = 300, overlap: int = 50):
        """
        Initialize the text chunker
        
        Args:
            chunk_size: Maximum number of characters per chunk
            overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._sentence_pattern = re.compile(r'[.!?](?:\s|$)')
        self._whitespace_pattern = re.compile(r'\s+')
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove excessive whitespace and newlines
        text = self._whitespace_pattern.sub(' ', text)
        return text.strip()
    
    def split_by_sentences(self, text: str) -> List[str]:
        """Split text into sentences using sentence patterns"""
        sentences = self._sentence_pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk_text(self, text: str) -> List[ChunkData]:
        """
        Split text into overlapping chunks
        
        Returns:
            List of ChunkData objects containing chunk text and metadata
        """
        if not text or len(text) <= self.chunk_size:
            return [ChunkData(text=self.clean_text(text), chunk_id=0, content_type=ContentType.DESCRIPTION)]
        
        text = self.clean_text(text)
        chunks = []
        
        # Try to split by sentences first for better semantic coherence
        sentences = self.split_by_sentences(text)
        
        if len(sentences) > 1:
            chunks = self._chunk_by_sentences(sentences)
        else:
            chunks = self._chunk_by_characters(text)
        
        return chunks
    
    def _chunk_by_sentences(self, sentences: List[str]) -> List[ChunkData]:
        """Group sentences into chunks"""
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size, save current chunk
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(ChunkData(
                    text=current_chunk.strip(),
                    chunk_id=chunk_id,
                    content_type=ContentType.DESCRIPTION
                ))
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-self.overlap:] if len(current_chunk) > self.overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
                chunk_id += 1
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(ChunkData(
                text=current_chunk.strip(),
                chunk_id=chunk_id,
                content_type=ContentType.DESCRIPTION
            ))
        
        return chunks
    
    def _chunk_by_characters(self, text: str) -> List[ChunkData]:
        """Fallback to character-based chunking"""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunk_text = text[i:i + self.chunk_size]
            chunks.append(ChunkData(
                text=chunk_text,
                chunk_id=i // (self.chunk_size - self.overlap),
                content_type=ContentType.DESCRIPTION
            ))
        return chunks

