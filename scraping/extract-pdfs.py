import asyncio
import aiohttp
import pdfplumber
import os
from openai import OpenAI
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from bs4 import BeautifulSoup

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

async def main():
    browser_config = BrowserConfig()  # Default browser configuration
    run_config = CrawlerRunConfig(
        remove_overlay_elements=True,
    )  

    with open("scraping/scraping_log.txt", "w", encoding="utf-8") as log_file:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(
                url="https://www.edpb.europa.eu/news/news/2025/edpb-publishes-csc-biannual-report_en",
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

if __name__ == "__main__":
    asyncio.run(main())