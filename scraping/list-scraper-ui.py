import requests
from bs4 import BeautifulSoup
import streamlit as st

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

st.title("Web Scraper UI")

base_url = st.text_input("Base URL", "https://www.imy.se")
listing_endpoint = st.text_input("Listing Endpoint", "tillsyner")
pagination_format = st.text_input("Pagination Format", "?query=&page=")
num_pages = st.number_input("Number of Pages", min_value=1, value=3, step=1)

if st.button("Scrape Data"):
    items = get_all_items(base_url, listing_endpoint, pagination_format, num_pages)
    for title, link in items:
        st.write(link)
