import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin, urlparse, parse_qs

# Function to extract all links from a page
def get_links_from_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract all 'href' links
    links = [a['href'] for a in soup.find_all('a', href=True)]
    return links

# Function to extract PDF links from a page
def get_pdfs_from_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all links that end with .pdf
    pdf_links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].lower().endswith('.pdf')]
    return pdf_links

# Function to find the next page link for pagination
def get_next_page_url(soup, base_url):
    next_page = soup.find('a', text='Next')  # Adjust this based on the actual button text or class
    if next_page and 'href' in next_page.attrs:
        return urljoin(base_url, next_page['href'])
    return None

# Scraping function with pagination handling
def scrape_pdfs(main_page_url, output_file='pdf_links.txt'):
    pdfs = set()
    page_number = 1  # Track page number
    current_page_url = main_page_url  # Start with the main page

    with open(output_file, 'w') as file:
        while current_page_url:
            print(f"Processing Page {page_number}: {current_page_url}")

            # Request the page and parse it
            response = requests.get(current_page_url)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract all links from the current page
            page_links = get_links_from_page(current_page_url)

            # Extract PDFs from each listed link on the page
            for link in page_links:
                absolute_url = urljoin(current_page_url, link)
                pdf_urls = get_pdfs_from_page(absolute_url)

                # Write new PDFs to file
                for pdf_url in pdf_urls:
                    if pdf_url not in pdfs:
                        pdfs.add(pdf_url)
                        file.write(pdf_url + '\n')
                        print(f"Found PDF: {pdf_url}")

            # Find next page URL
            current_page_url = get_next_page_url(soup, main_page_url)
            if current_page_url:
                page_number += 1

# Example usage
main_page_url = 'https://www.edpb.europa.eu/news/news_en'  # Replace with actual website
scrape_pdfs(main_page_url)
