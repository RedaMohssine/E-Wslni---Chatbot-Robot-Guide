"""
Semantic (header-aware) chunking for markdown documents.
Splits at H1/H2 boundaries, merges small sections, sub-splits large ones.
"""

import json
import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = "data"
MAX_CHUNK_SIZE = 1500
MIN_CHUNK_SIZE = 100
FALLBACK_OVERLAP = 200


def split_by_headers(text):
    """Split text at markdown H1/H2 headers into sections."""
    header_pattern = re.compile(r'^(#{1,2})\s+(.+)', re.MULTILINE)
    
    sections = []
    matches = list(header_pattern.finditer(text))
    
    if not matches:
        # No headers found — return the whole text as one section
        return [{"header": "", "text": text.strip()}]
    
    # Add text BEFORE the first header (if any)
    if matches[0].start() > 0:
        pre_text = text[:matches[0].start()].strip()
        if pre_text:
            sections.append({"header": "", "text": pre_text})
    
    # Split at each header
    for i, match in enumerate(matches):
        header_level = len(match.group(1))  # 1 for #, 2 for ##
        header_text = match.group(2).strip()
        
        # Get the content between this header and the next
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        
        sections.append({
            "header": header_text,
            "header_level": header_level,
            "text": section_text,
        })
    
    return sections


def merge_small_sections(sections, min_size=MIN_CHUNK_SIZE):
    """Merge sections smaller than min_size with their neighbors."""
    if not sections:
        return sections
    
    merged = []
    buffer = None
    
    for section in sections:
        if buffer is None:
            if len(section["text"]) < min_size:
                buffer = section  # Too small, start buffering
            else:
                merged.append(section)
        else:
            # Merge buffer with current section
            combined_text = buffer["text"] + "\n\n" + section["text"]
            combined = {
                "header": buffer["header"] or section["header"],
                "text": combined_text,
            }
            if len(combined_text) < min_size:
                buffer = combined  # Still too small, keep buffering
            else:
                merged.append(combined)
                buffer = None
    
    # Don't forget the last buffer
    if buffer:
        if merged:
            # Merge with the last section
            last = merged[-1]
            last["text"] = last["text"] + "\n\n" + buffer["text"]
        else:
            merged.append(buffer)
    
    return merged


def subsplit_large_sections(sections, max_size=MAX_CHUNK_SIZE, overlap=FALLBACK_OVERLAP):
    """Sub-split sections exceeding max_size, preserving header context."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    result = []
    for section in sections:
        if len(section["text"]) <= max_size:
            result.append(section)
        else:
            # Sub-split, but prefix each sub-chunk with the header
            header_prefix = f"# {section['header']}\n\n" if section["header"] else ""
            sub_chunks = splitter.split_text(section["text"])
            
            for j, sub_text in enumerate(sub_chunks):
                # Add header prefix if not already present
                if header_prefix and not sub_text.startswith("#"):
                    sub_text = header_prefix + sub_text
                
                result.append({
                    "header": section["header"],
                    "text": sub_text,
                    "sub_chunk": j + 1,
                    "total_sub_chunks": len(sub_chunks),
                })
    
    return result


def semantic_chunk_text(text, source_name=""):
    """Full chunking pipeline: split by headers, merge small, sub-split large."""
    sections = split_by_headers(text)
    sections = merge_small_sections(sections)
    sections = subsplit_large_sections(sections)
    return sections


def load_json_data(filepath):
    """Load scraped website JSON data."""
    print(f"  Loading JSON: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    documents = []
    for item in data:
        text = f"# {item['title']}\n\n{item['content']}"
        metadata = {
            "source": item.get("url", "unknown"),
            "title": item.get("title", "unknown"),
            "type": "website"
        }
        documents.append({"text": text, "metadata": metadata})
    
    print(f"  Loaded {len(documents)} web pages")
    return documents


def load_markdown_data(filepath):
    """Load a single Markdown file."""
    filename = os.path.basename(filepath)
    print(f"  Loading Markdown: {filename}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    
    metadata = {
        "source": filename,
        "title": filename.replace(".md", ""),
        "type": "pdf_parsed"
    }
    
    print(f"  Loaded {len(text)} characters")
    return [{"text": text, "metadata": metadata}]


def chunk_documents_semantic(documents):
    """Apply semantic chunking to all documents and attach metadata."""
    print(f"  Chunking {len(documents)} documents...")
    
    all_chunks = []
    
    for doc in documents:
        sections = semantic_chunk_text(doc["text"], doc["metadata"].get("source", ""))
        
        for i, section in enumerate(sections):
            chunk = {
                "text": section["text"],
                "metadata": {
                    **doc["metadata"],
                    "chunk_index": i,
                    "total_chunks": len(sections),
                    "section_header": section.get("header", ""),
                }
            }
            all_chunks.append(chunk)
    
    print(f"  Created {len(all_chunks)} semantic chunks")
    return all_chunks


def load_all_data():
    """Load all data sources and apply semantic chunking."""
    print("=" * 60)
    print("LOADING & SEMANTIC CHUNKING")
    print("=" * 60)
    
    all_documents = []
    
    # Load JSON
    json_path = os.path.join(DATA_DIR, "emines_docs.json")
    if os.path.exists(json_path):
        all_documents.extend(load_json_data(json_path))
    
    # Load Markdown files
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".md"):
            md_path = os.path.join(DATA_DIR, filename)
            all_documents.extend(load_markdown_data(md_path))
    
    # Apply semantic chunking
    chunks = chunk_documents_semantic(all_documents)
    
    sizes = [len(c["text"]) for c in chunks]
    avg_size = sum(sizes) / len(sizes)
    print(f"  Documents: {len(all_documents)} | Chunks: {len(chunks)}")
    print(f"  Avg size: {avg_size:.0f} chars | Range: {min(sizes)}-{max(sizes)} chars")
    
    return chunks


if __name__ == "__main__":
    chunks = load_all_data()
    print(f"\nDone. {len(chunks)} chunks ready.")
