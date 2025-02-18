import streamlit as st
import requests
from bs4 import BeautifulSoup
import asyncio
import aiohttp
import pdfplumber
import os
import logging
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from openai import OpenAI
import sys

# ----------------- Logging Setup -----------------
logging.basicConfig(level=logging.INFO)

# ----------------- Web Scraper Functions -----------------

def get_page_items(url, base_url, listing_endpoint):
    """Extracts all item links from a page."""
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code != 200:
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        items = set()

        for item in soup.find_all("a"):
            link = item.get("href")
            title = item.text.strip()
            if link and f"/{listing_endpoint}/" in link and link != url and not link.endswith("/rss"):
                if not link.startswith("http"):
                    link = base_url + link
                items.add((title, link))

        return list(items)
    except Exception as e:
        logging.error(f"Error scraping {url}: {e}")
        return []


def get_all_items(base_url, listing_endpoint, pagination_format, num_pages):
    """Scrapes multiple pages to collect all links."""
    all_items = set()

    for page in range(1, num_pages + 1):
        url = f"{base_url}/{listing_endpoint}/{pagination_format}{page}"
        items = get_page_items(url, base_url, listing_endpoint)
        if not items:
            break
        
        all_items.update(items)
        st.write(f"Scraped page {page}")

    return list(all_items)


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF {pdf_path}: {e}")
        return ""


def summarize_text(text):
    """Summarizes extracted text using OpenAI."""
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Summarize the following text into a concise paragraph."},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error in summarization: {e}")
        return "Summarization failed."


async def download_pdf(url, session, save_path):
    """Downloads a PDF file asynchronously."""
    try:
        async with session.get(url) as response:
            if response.status == 200:
                with open(save_path, 'wb') as f:
                    f.write(await response.read())
                return save_path
    except Exception as e:
        logging.error(f"Error downloading {url}: {e}")
    return None


async def scraper(url):
    """Scrapes a webpage, extracts text, finds PDFs, and summarizes content."""
    # try:
    browser_config = BrowserConfig(
        browser_type="chromium",  # Use Chromium for compatibility
        headless=True,  # Run in headless mode for Streamlit
        use_managed_browser=False,  # Disable managed mode to prevent conflicts
        debugging_port=None,  # No debugging port needed
        proxy=None,  # Disable proxy unless explicitly required
        text_mode=True,  # Optimize for text scraping (faster)
        light_mode=True,  # Further performance optimizations
        verbose=True,  # Enable logging for debugging
        ignore_https_errors=True,  # Avoid SSL certificate issues
        java_script_enabled=True  # Enable JS for dynamic content
    )

    run_config = CrawlerRunConfig(remove_overlay_elements=True)

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=run_config)

        # Extract Text from Page
        soup = BeautifulSoup(result.html, "html.parser")
        paragraphs = "\n".join([p.get_text() for p in soup.find_all("p")])
        summarized_text = summarize_text(paragraphs) if paragraphs else "No text available to summarize."

        # Extract PDFs
        internal_links = result.links.get("internal", [])
        pdf_links = [link['href'] for link in internal_links if '.pdf' in link['href'].lower()]

        # Download PDFs
        extracted_texts = []
        if pdf_links:
            async with aiohttp.ClientSession() as session:
                for i, link in enumerate(pdf_links):
                    pdf_path = f"document_{i}.pdf"
                    saved_path = await download_pdf(link, session, pdf_path)
                    if saved_path:
                        extracted_texts.append(extract_text_from_pdf(saved_path))

        return summarized_text, pdf_links, extracted_texts
    # except Exception as e:
    #     logging.error(f"Scraping error: {e}")
    #     return "Scraping failed.", [], []


# ----------------- Streamlit UI -----------------

st.title("Web Scraper with AI Summarization")

base_url = st.text_input("Enter Base URL", "https://www.imy.se")
listing_endpoint = st.text_input("Enter Listing Endpoint", "tillsyner")
pagination_format = st.text_input("Enter Pagination Format", "?query=&page=")
num_pages = st.number_input("Enter Number of Pages", 1, 20, 3)

async def run_scraper():
    """Runs all scraper tasks asynchronously."""
    tasks = [scraper(link) for _, link in items]  # Create async tasks
    results = await asyncio.gather(*tasks)  # Run all tasks concurrently
    return results

if st.button("Start Scraping"):
    
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    with st.spinner("Scraping in progress..."):
        items = get_all_items(base_url, listing_endpoint, pagination_format, num_pages)

    if items:
        st.success(f"Found {len(items)} items!")
        for title, link in items:
            st.write(link)

        scrape_results = []
        # Run scraper asynchronously
        with st.spinner("Scraping in progress..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            scrape_results = asyncio.run(run_scraper())

        # Display Results
        for i, (summary, pdf_links, extracted_texts) in enumerate(scrape_results):
            st.subheader(f"Result {i+1}")
            st.write("**Summarized Text:**", summary)
            if pdf_links:
                st.write("**Extracted PDFs:**")
                for pdf in pdf_links:
                    st.markdown(f"[Download PDF]({pdf})")
            if extracted_texts:
                st.write("**Extracted Text from PDFs:**")
                st.text(extracted_texts[0])
    else:
        st.warning("No items found.")