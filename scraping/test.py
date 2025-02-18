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
            url="https://www.imy.se/tillsyner/",
            config=run_config
        )

        internal_links = result.links.get("internal", [])
        # external_links = result.links.get("external", [])
        print(internal_links[0])

        # # Save the markdown content into a text file
        with open("output.txt", "w", encoding="utf-8") as file:
            for link in internal_links:
                file.write(link['href'] + '\n')
        
        print("Markdown content saved to output.txt")

    
if __name__ == "__main__":
    asyncio.run(main())
