import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin

# Function to extract all links from the main page
def get_links_from_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract all 'href' links
    links = [a['href'] for a in soup.find_all('a', href=True)]
    return links

# Function to extract PDF links from a given page
def get_pdfs_from_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all links that end with .pdf (generalized approach)
    pdf_links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].lower().endswith('.pdf')]
    return pdf_links

# Main scraping function that saves PDF links to a text file
def scrape_pdfs(main_page_url, output_file='pdf_links.txt'):
    # Get all links from the main page
    page_links = get_links_from_page(main_page_url)
    
    pdfs = set()
    # Open the output file for writing PDF links
    with open(output_file, 'w') as file:
        # Go through each link and extract PDF links
        for link in page_links:
            # Convert relative links to absolute URLs
            absolute_url = urljoin(main_page_url, link)
            
            # Extract PDFs from the linked page
            pdf_urls = get_pdfs_from_page(absolute_url)
            
            # Write each found PDF URL to the text file
            for pdf_url in pdf_urls:
                size = len(pdfs)
                pdfs.add(pdf_url)
                if len(pdfs) != size:
                    file.write(pdf_url + '\n')
                    print(f"Found PDF: {pdf_url}")

# Example usage
main_page_url = 'https://www.imy.se/tillsyner/'  # Replace with actual main page URL
scrape_pdfs(main_page_url)
