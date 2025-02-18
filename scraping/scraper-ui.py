import requests
from bs4 import BeautifulSoup
import streamlit as st
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from crawl4ai import AsyncWebCrawler
from openai import OpenAI
import aiohttp
import pdfplumber
import asyncio
import sys
import nest_asyncio

nest_asyncio.apply()

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def get_page_items(url, base_url, listing_endpoint):
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    if response.status_code != 200:
        st.error(f"Failed to fetch {url}")
        return []
    
    soup = BeautifulSoup(response.text, "html.parser")
    items = set()
    
    # Find the items in the list
    for item in soup.find_all("a"):
        link = item.get("href")
        title = item.text.strip()
        if link and f"/{listing_endpoint}/" in link and link != url and not link.endswith("/rss"):
            if not link.startswith("http"):
                link = base_url + link
            items.add((title, link))
    
    return list(items)

def get_all_items(base_url, listing_endpoint, pagination_format, num_pages):
    all_items = set()
    
    for page in range(1, num_pages + 1):
        url = f"{base_url}/{listing_endpoint}/{pagination_format}{page}"
        items = get_page_items(url, base_url, listing_endpoint)
        if not items:
            break  # Stop when no more items are found
        
        all_items.update(items)
        st.write(f"Scraped page {page}")
    
    return list(all_items)


def extract_text_from_pdf(pdf_path, text_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    with open(text_path, "w", encoding="utf-8") as text_file:
        text_file.write(text)

def summarize_text(text):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Summarize the following text into a concise and informative paragraph."},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content

async def download_pdf(url, session, save_path, log_file):
    async with session.get(url) as response:
        if response.status == 200:
            with open(save_path, 'wb') as f:
                f.write(await response.read())
            log_file.write(f"Downloaded: {url}\n")
        else:
            log_file.write(f"Failed to download: {url}\n")

async def scrap_pages(urls):
    browser_config = BrowserConfig()  # Default browser configuration
    run_config = CrawlerRunConfig(
        remove_overlay_elements=True,
    )  

    with open("scraping/scraping_log.txt", "w", encoding="utf-8") as log_file:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            for url in urls:
                result = await crawler.arun(
                    url=url,
                    config=run_config
                )

                internal_links = result.links.get("internal", [])

                # Extract paragraphs from the page
                soup = BeautifulSoup(result.html, "html.parser")
                paragraphs = "\n".join([p.get_text() for p in soup.find_all("p")])
                summarized_text = summarize_text(paragraphs) if paragraphs else "No text available to summarize."

                # Filter links that contain '.pdf'
                pdf_links = [link['href'] for link in internal_links if '.pdf' in link['href'].lower()]

                log_file.write(f"Extracted Paragraphs:\n{paragraphs}\n\n")
                log_file.write(f"Summarized Text:\n{summarized_text}\n\n")
                log_file.write(f"PDF Links:\n{chr(10).join(pdf_links)}\n\n")

                if pdf_links:
                    async with aiohttp.ClientSession() as session:
                        for i, link in enumerate(pdf_links):
                            pdf_path = f"scraping/pdfs/document_{i}.pdf"
                            text_path = f"scraping/extracted_texts/document_{i}.txt"
                            await download_pdf(link, session, pdf_path, log_file)
                            extract_text_from_pdf(pdf_path, text_path)
                else:
                    log_file.write("No PDF links found.\n")

st.title("Web Scraper UI")

base_url = st.text_input("Base URL", "https://www.imy.se")
listing_endpoint = st.text_input("Listing Endpoint", "tillsyner")
pagination_format = st.text_input("Pagination Format", "?query=&page=")
num_pages = st.number_input("Number of Pages", min_value=1, value=3, step=1)

if st.button("Scrape Data"):
    items = get_all_items(base_url, listing_endpoint, pagination_format, num_pages)
    for title, link in items:
        st.write(link)

    asyncio.run(scrap_pages([link for _, link in items]))