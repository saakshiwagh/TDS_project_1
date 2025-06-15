#!/usr/bin/env python3
import os
import json
import time
import hashlib
import mimetypes
import requests
from tqdm import tqdm
from datetime import datetime, timezone
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()


# -- CONFIG ---------------------------------------------------------------


DISCOURSE_BASE = "https://discourse.onlinedegree.iitm.ac.in"
SESSION_COOKIE = os.getenv("DISCOURSE_SESSION")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("API_KEY")
COOKIES = {"_t":SESSION_COOKIE,}

START_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2025, 4, 14, 23, 59, 59, tzinfo=timezone.utc)

MAX_RETRIES = 5
RETRY_BACKOFF = [1, 2, 4, 8, 16]
CACHE_FILE = "gemini_image_cache.json"
OUTPUT_FILE = "all_discourse_posts_enriched.json"

# -- CACHE ---------------------------------------------------------------
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        image_caption_cache = json.load(f)
else:
    image_caption_cache = {}

def save_cache():
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(image_caption_cache, f, indent=2, ensure_ascii=False)

def hash_url(url):
    return hashlib.md5(url.encode()).hexdigest()

def describe_image(image_url):
    key = hash_url(image_url)
    if key in image_caption_cache:
        return image_caption_cache[key]

    try:
        image_bytes = requests.get(image_url, timeout=10).content
        mime_type, _ = mimetypes.guess_type(image_url)
        image = types.Part.from_bytes(
            data=image_bytes, mime_type=mime_type or "image/jpeg"
        )
        client = genai.Client(api_key=GOOGLE_API_KEY)
        for attempt in range(MAX_RETRIES):
            try:
                response = client.models.generate_content(
                    model="gemini-1.5-flash",
                    contents=["Describe this image in a student context.", image],
                )
                caption = response.text.strip()
                image_caption_cache[key] = caption
                save_cache()
                return caption
            except Exception as e:
                if "429" in str(e).lower():
                    wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                    print(f"⚠️ Rate limit, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"❌ Gemini error: {e}")
                    break
        return "[Gemini caption failed]"
    except Exception as e:
        return f"[Image error: {e}]"

# -- RESUME --------------------------------------------------------------
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        all_posts = json.load(f)
    scraped_ids = {post["id"] for post in all_posts}
else:
    all_posts = []
    scraped_ids = set()


def save_topic_progress(topics_completed, fname="topics_done.txt"):
    with open(fname, "w") as f:
        f.write(str(topics_completed))

# -- SCRAPE A TOPIC FUNCTION ---------------------------------------------
def fetch_and_add_topic(topic_id, topic_slug):
    print(f"\n🔎 Fetching and ensuring topic: {topic_slug} ({topic_id})")
    page = 1
    added_count = 0
    while True:
        turl = f"{DISCOURSE_BASE}/t/{topic_slug}/{topic_id}.json?page={page}"
        try:
            resp = requests.get(turl, cookies=COOKIES, timeout=10)
            if resp.status_code != 200:
                break
            data = resp.json()
        except Exception as e:
            print(f"❌ Error fetching topic {topic_id}: {e}")
            break

        posts = data.get("post_stream", {}).get("posts", [])
        if not posts:
            break

        for post in posts:
            if post["id"] in scraped_ids:
                continue

            soup = BeautifulSoup(post["cooked"], "html.parser")
            text = soup.get_text(separator="\n", strip=True)

            img_urls = set()
            for a in soup.find_all("a", class_="lightbox"):
                href = a.get("href")
                if href:
                    img_urls.add(href if href.startswith("http") else DISCOURSE_BASE + href)
            for img in soup.find_all("img"):
                src = img.get("src")
                if src and not any(cls in (img.get("class") or []) for cls in ["emoji", "avatar"]):
                    img_urls.add(src if src.startswith("http") else DISCOURSE_BASE + src)

            descriptions = []
            for img_url in img_urls:
                caption = describe_image(img_url)
                print(f"🖼 {img_url}\n📝 {caption}")
                descriptions.append(f"[Image] {img_url}\n[Caption] {caption}")

            post_text = text + "\n\n" + "\n\n".join(descriptions) if descriptions else text

            all_posts.append({
                "id": post["id"],
                "username": post["username"],
                "created_at": post["created_at"],
                "url": f"{DISCOURSE_BASE}/t/{topic_slug}/{topic_id}/{post['post_number']}",
                "post": post_text,
                "images": list(img_urls)
            })
            scraped_ids.add(post["id"])
            added_count += 1

            # Save after every post for safety
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(all_posts, f, indent=2, ensure_ascii=False)

        page += 1
    if added_count:
        print(f"✅ Added {added_count} new posts from {topic_slug}")
    else:
        print(f"Topic {topic_slug} was already complete or had no new posts.")

# -- MAIN SCRAPING LOOP --------------------------------------------------
topic_list = []
for page in range(30):
    url = f"{DISCOURSE_BASE}/c/courses/tds-kb/34.json?page={page}"
    try:
        resp = requests.get(url, cookies=COOKIES, timeout=10)
        resp.raise_for_status()
        for topic in resp.json()["topic_list"]["topics"]:
            created = datetime.fromisoformat(topic["created_at"].replace("Z", "+00:00"))
            if START_DATE <= created <= END_DATE:
                topic_list.append({"id": topic["id"], "slug": topic["slug"]})
    except Exception as e:
        print(f"❌ Failed to fetch page {page}: {e}")
        break

print(f"✅ Found {len(topic_list)} topics")

for topic in tqdm(topic_list, desc="Scraping Topics"):
    page = 1
    while True:
        turl = f"{DISCOURSE_BASE}/t/{topic['slug']}/{topic['id']}.json?page={page}"
        try:
            resp = requests.get(turl, cookies=COOKIES, timeout=10)
            if resp.status_code != 200:
                break
            data = resp.json()
        except Exception as e:
            print(f"❌ Error fetching topic {topic['id']}: {e}")
            break

        posts = data.get("post_stream", {}).get("posts", [])
        if not posts:
            break

        for post in posts:
            if post["id"] in scraped_ids:
                continue

            soup = BeautifulSoup(post["cooked"], "html.parser")
            text = soup.get_text(separator="\n", strip=True)

            img_urls = set()
            for a in soup.find_all("a", class_="lightbox"):
                href = a.get("href")
                if href:
                    img_urls.add(href if href.startswith("http") else DISCOURSE_BASE + href)
            for img in soup.find_all("img"):
                src = img.get("src")
                if src and not any(cls in (img.get("class") or []) for cls in ["emoji", "avatar"]):
                    img_urls.add(src if src.startswith("http") else DISCOURSE_BASE + src)

            descriptions = []
            for img_url in img_urls:
                caption = describe_image(img_url)
                print(f"🖼 {img_url}\n📝 {caption}")
                descriptions.append(f"[Image] {img_url}\n[Caption] {caption}")

            post_text = text + "\n\n" + "\n\n".join(descriptions) if descriptions else text

            all_posts.append({
                "id": post["id"],
                "username": post["username"],
                "created_at": post["created_at"],
                "url": f"{DISCOURSE_BASE}/t/{topic['slug']}/{topic['id']}/{post['post_number']}",
                "post": post_text,
                "images": list(img_urls)
            })
            scraped_ids.add(post["id"])

            # Save after every post for safety
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(all_posts, f, indent=2, ensure_ascii=False)

        page += 1

    topics_completed = topic_list.index(topic) + 1
    save_topic_progress(topics_completed)

print(f"✅ Scraped {len(all_posts)} posts → {OUTPUT_FILE}")

# -- ENSURE MUST-HAVE TOPICS ---------------------------------------------
must_have_topics = [
    (155939, "ga5-question-8-clarification"),  # Add more as needed
]
for tid, slug in must_have_topics:
    fetch_and_add_topic(tid, slug)
print(f"✅ All must-have topics processed and saved to {OUTPUT_FILE}")