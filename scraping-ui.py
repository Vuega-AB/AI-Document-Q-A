import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import streamlit as st

# Function to extract all links from a page
def get_links_from_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    links = [a['href'] for a in soup.find_all('a', href=True)]
    return links

# Function to extract PDF links from a page
def get_pdfs_from_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    pdf_links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].lower().endswith('.pdf')]
    return pdf_links

# Function to find the next page dynamically
def get_next_page_url(soup, current_url):
    next_page = None
    pagination = soup.find("nav", class_="pagination") or soup.find("ul", class_="pagination") or soup.find("div", class_="pagination")

    if pagination:
        active_page = pagination.find("li", class_="active")
        if active_page and active_page.find_next_sibling("li"):
            next_link = active_page.find_next_sibling("li").find("a", href=True)
            if next_link:
                next_page = urljoin(current_url, next_link["href"])
    
    if not next_page:
        next_buttons = soup.find_all("a", href=True)
        for btn in next_buttons:
            btn_text = btn.get_text(strip=True).lower()
            if btn_text in ["next", ">", "»", "nästa"]:
                next_page = urljoin(current_url, btn["href"])
                break

    return next_page

# Scraping function with pagination handling
def scrape_pdfs(base_url, max_pages=50):
    pdfs = set()
    current_page_url = base_url
    page_count = 0
    
    st.write(f"**Starting PDF extraction from:** {base_url}")
    
    while current_page_url:
        page_count += 1
        st.write(f"Processing Page {page_count}: {current_page_url}")
        
        response = requests.get(current_page_url)
        if response.status_code != 200:
            st.error(f"Stopping: Page {page_count} returned {response.status_code}")
            break
        
        soup = BeautifulSoup(response.text, 'html.parser')
        page_links = get_links_from_page(current_page_url)
        
        for link in page_links:
            absolute_url = urljoin(current_page_url, link)
            pdf_urls = get_pdfs_from_page(absolute_url)
            
            for pdf_url in pdf_urls:
                if pdf_url not in pdfs:
                    pdfs.add(pdf_url)
                    st.write(f"✅ Found PDF: {pdf_url}")
        
        current_page_url = get_next_page_url(soup, current_page_url)
        
        if not current_page_url or page_count >= max_pages:
            st.success("No more pages found or reached max pages. Stopping.")
            break
    
    return pdfs

# Streamlit UI
st.title("📄 PDF Scraper with Pagination")
input_url = st.text_input("Enter the URL to scrape:")

if st.button("Start Scraping") and input_url:
    pdf_results = scrape_pdfs(input_url)
    
    if pdf_results:
        st.subheader("Extracted PDF Links:")
        for pdf in pdf_results:
            st.write(pdf)
    else:
        st.warning("No PDFs found.")