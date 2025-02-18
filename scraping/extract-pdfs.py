import asyncio
import aiohttp
import pdfplumber
import os
from openai import OpenAI

from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()
OpenAI_KEY = os.getenv("OPENAI_API_KEY")

async def download_pdf(url, session, save_path):
    async with session.get(url) as response:
        if response.status == 200:
            with open(save_path, 'wb') as f:
                f.write(await response.read())
            print(f"Downloaded: {url}")
        else:
            print(f"Failed to download: {url}")

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text

def summarize_text(text):
    client = OpenAI(api_key=OpenAI_KEY)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Summarize the following text into a concise and informative paragraph. Don't change the language of the text."},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content

async def main():
    browser_config = BrowserConfig()  # Default browser configuration
    run_config = CrawlerRunConfig(
        remove_overlay_elements=True,
    )  

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url="https://www.imy.se/tillsyner/",
            config=run_config
        )

        internal_links = result.links.get("internal", [])
        
        # Extract paragraphs from the page
        soup = BeautifulSoup(result.html, "html.parser")
        paragraphs = "\n".join([p.get_text() for p in soup.find_all("p")])
        summarized_text = summarize_text(paragraphs) if paragraphs else "No text available to summarize."
        
        # Filter links that contain '.pdf'
        pdf_links = [link['href'] for link in internal_links if '.pdf' in link['href'].lower()]
        
        if pdf_links:
            os.makedirs("pdfs", exist_ok=True)
            os.makedirs("extracted_texts", exist_ok=True)
            async with aiohttp.ClientSession() as session:
                for i, link in enumerate(pdf_links):
                    pdf_path = f"pdfs/document_{i}.pdf"
                    await download_pdf(link, session, pdf_path)
                    extracted_text = extract_text_from_pdf(pdf_path)
                    with open(f"extracted_texts/document_{i}.txt", "w", encoding="utf-8") as text_file:
                        text_file.write(extracted_text)
            print("PDFs downloaded and text extracted.")
        else:
            print("No PDF links found.")
        
        # Save summarized paragraphs to a file
        with open("scraping/paragraphs_summary.txt", "w", encoding="utf-8") as file:
            file.write(summarized_text)
        print("Summarized paragraphs saved to paragraphs_summary.txt")
    
if __name__ == "__main__":
    asyncio.run(main())
