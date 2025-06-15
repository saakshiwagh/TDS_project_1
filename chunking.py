import os
import json
import re
import glob
import urllib.parse
from bs4 import BeautifulSoup

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def clean_html(text):
    if not text:
        return ""
    soup = BeautifulSoup(text, 'html.parser')
    for tag in soup(["script", "style"]): tag.decompose()
    text = soup.get_text(separator='\n')
    return re.sub(r'\s+', ' ', text).strip()

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text: return []
    # Normalize whitespace, prefer paragraph and sentence boundaries
    text = re.sub(r'\n+', '\n', text).strip()
    paragraphs = text.split('\n')
    chunks, current = [], ""
    for para in paragraphs:
        para = para.strip()
        if not para: continue
        if len(current) + len(para) + 1 <= chunk_size:
            current += (" " if current else "") + para
        else:
            if current:
                chunks.append(current)
            current = para
    if current:
        chunks.append(current)
    # Now apply overlap
    overlapped = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            overlapped.append(chunk)
        else:
            prev = overlapped[-1]
            overlap = prev[-CHUNK_OVERLAP:]
            merged = overlap + " " + chunk if overlap not in chunk else chunk
            overlapped.append(merged)
    return overlapped

def discourse_chunker(discourse_json, out_file):
    with open(discourse_json, "r", encoding="utf-8") as f, open(out_file, "w", encoding="utf-8") as out:
        posts = json.load(f)
        for post in posts:
            text = clean_html(post.get("post", ""))
            url = post.get("url")
            post_id = post.get("id")
            if len(text) < 20: continue
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                out.write(json.dumps({
                    "id": f"discourse_{post_id}_{i}",
                    "content": chunk,
                    "source": url,
                    "type": "discourse_post"
                }, ensure_ascii=False) + "\n")

def slugify_heading(heading):
    # Remove '#' and spaces, lower, replace spaces & special chars with '-'
    anchor = re.sub(r'[^a-z0-9]+', '-', heading.strip("# ").lower())
    anchor = anchor.strip('-')
    return anchor

def markdown_chunker(md_folder, out_file):
    md_files = glob.glob(os.path.join(md_folder, "*.md"))
    with open(out_file, "a", encoding="utf-8") as out:
        for md_file in md_files:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
            filename = os.path.splitext(os.path.basename(md_file))[0]
            lines = content.splitlines()
            section, heading = [], None
            for line in lines:
                if line.strip().startswith("#"):
                    if section:
                        text = "\n".join(section).strip()
                        chunks = chunk_text(text)
                        slug = slugify_heading(heading) if heading else ""
                        url = f"https://tds.s-anand.net/#/{filename}"
                        if slug:
                            url += f"?id={slug}"
                        for i, chunk in enumerate(chunks):
                            out.write(json.dumps({
                                "id": f"{filename}_{slug}_{i}",
                                "content": chunk,
                                "source": url,
                                "type": "course_content"
                            }, ensure_ascii=False) + "\n")
                    heading = line.strip()
                    section = [line]
                else:
                    section.append(line)
            # last section
            if section:
                text = "\n".join(section).strip()
                chunks = chunk_text(text)
                slug = slugify_heading(heading) if heading else ""
                url = f"https://tds.s-anand.net/#/{filename}"
                if slug:
                    url += f"?id={slug}"
                for i, chunk in enumerate(chunks):
                    out.write(json.dumps({
                        "id": f"{filename}_{slug}_{i}",
                        "content": chunk,
                        "source": url,
                        "type": "course_content"
                    }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    discourse_chunker("all_discourse_posts_enriched.json", "chunks.jsonl")
    markdown_chunker("course_content_md", "chunks.jsonl")
    print("✅ Chunks created (structure-aware, overlapped) in chunks.jsonl")
