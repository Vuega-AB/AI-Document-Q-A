import asyncio

from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig

async def main():
    browser_config = BrowserConfig()  # Default browser configuration
    run_config = CrawlerRunConfig(
        remove_overlay_elements=True,
    )  

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url="https://www.imy.se/tillsyner/bonnier-news-ab/",
            config=run_config
        )

        internal_links = result.links.get("internal", [])
        paragraphs = result  # Extract paragraphs from the page
        
        # Filter links that contain '.pdf'
        pdf_links = [link['href'] for link in internal_links if '.pdf' in link['href'].lower()]
        
        if pdf_links:
            with open("scraping\pdf_links.txt", "w", encoding="utf-8") as file:
                for link in pdf_links:
                    file.write(link + '\n')
            print("PDF links saved to pdf_links.txt")
        else:
            print("No PDF links found.")
        
        # Save paragraphs to a file
        with open("scraping\paragraphs.txt", "w", encoding="utf-8") as file:
            file.write(paragraphs)
        print("Paragraphs saved to paragraphs.txt")
    
if __name__ == "__main__":
    asyncio.run(main())