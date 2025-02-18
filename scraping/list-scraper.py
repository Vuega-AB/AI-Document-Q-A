import requests
from bs4 import BeautifulSoup
import argparse

def get_page_items(url, base_url, listing_endpoint):
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    if response.status_code != 200:
        print(f"Failed to fetch {url}")
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
            break  # Stop when no more items are found
        
        all_items.update(items)
        print(f"Scraped page {page}")
    
    return list(all_items)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Web Scraper CLI")
    parser.add_argument("--base_url", type=str, required=True, help="Base URL of the website")
    parser.add_argument("--listing_endpoint", type=str, required=True, help="Listing endpoint to scrape")
    parser.add_argument("--pagination_format", type=str, default="?query=&page=", help="Pagination format")
    parser.add_argument("--num_pages", type=int, default=3, help="Number of pages to scrape")
    
    args = parser.parse_args()
    
    items = get_all_items(args.base_url, args.listing_endpoint, args.pagination_format, args.num_pages)
    
    for title, link in items:
        print(f"{title}: {link}")
