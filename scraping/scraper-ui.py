import streamlit as st
import requests
from bs4 import BeautifulSoup
import asyncio
import aiohttp
import pdfplumber
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from openai import OpenAI

# ----------------- Web Scraper Functions -----------------

def get_page_items(url, base_url, listing_endpoint):
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


def get_all_items(base_url, listing_endpoint, pagination_format, num_pages):
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
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text


def summarize_text(text):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Summarize the following text into a concise paragraph."},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content


async def download_pdf(url, session, save_path):
    async with session.get(url) as response:
        if response.status == 200:
            with open(save_path, 'wb') as f:
                f.write(await response.read())
            return save_path
    return None


async def scraper(url):
    browser_config = BrowserConfig()
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


# ----------------- Streamlit UI -----------------

st.title("Web Scraper with AI Summarization")

base_url = st.text_input("Enter Base URL", "https://www.imy.se")
listing_endpoint = st.text_input("Enter Listing Endpoint", "tillsyner")
pagination_format = st.text_input("Enter Pagination Format", "?query=&page=")
num_pages = st.number_input("Enter Number of Pages", 1, 20, 3)

# if st.button("Start Scraping"):
#     asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

#     with st.spinner("Scraping in progress..."):
#         items = get_all_items(base_url, listing_endpoint, pagination_format, num_pages)

#     if items:
#         st.success(f"Found {len(items)} items!")
#         for title, link in items:
#             st.write(link)
        
#         # Scrape Each Link Asynchronously
#         def run_scraper():
#             results = []
#             for _, link in items:
#                 loop = asyncio.new_event_loop()
#                 asyncio.set_event_loop(loop)
#                 result = loop.run_until_complete(scraper(link))  # ✅ Run each scrape in a separate loop
#                 results.append(result)
#                 loop.close()
#             return results

        
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
#         scrape_results = loop.run_until_complete(run_scraper())

#         # Display Results
#         for i, (summary, pdf_links, extracted_texts) in enumerate(scrape_results):
#             st.subheader(f"Result {i+1}")
#             st.write("**Summarized Text:**", summary)
#             if pdf_links:
#                 st.write("**Extracted PDFs:**")
#                 for pdf in pdf_links:
#                     st.markdown(f"[Download PDF]({pdf})")
#             if extracted_texts:
#                 st.write("**Extracted Text from PDFs:**")
#                 st.text(extracted_texts[0])

#     else:
#         st.warning("No items found.")

async def run_scraper():
    tasks = [scraper(link) for _, link in items]  # Create async tasks
    results = await asyncio.gather(*tasks)  # Run all tasks concurrently
    return results

if st.button("Start Scraping"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    with st.spinner("Scraping in progress..."):
        items = get_all_items(base_url, listing_endpoint, pagination_format, num_pages)

    if items:
        st.success(f"Found {len(items)} items!")
        for title, link in items:
            st.write(link)

        # Run scraper asynchronously
        scrape_results = asyncio.run(run_scraper())  # ✅ Properly runs async tasks
        
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
