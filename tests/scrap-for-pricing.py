import requests
from bs4 import BeautifulSoup
import re
import pandas as pd

# URL of Together AI pricing page
URL = "https://www.together.ai/pricing#inference"

def clean_price(price_text):
    """Removes extra text before the actual price starting with '$'."""
    match = re.search(r"\$\d+(\.\d+)?", price_text)  # Find first occurrence of $ followed by digits
    return match.group(0) if match else "Unknown Price"  # Return price if found, else 'Unknown Price'

def get_all_model_pricing():
    response = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"})
    if response.status_code != 200:
        return {"Error": "Failed to fetch pricing data"}
    
    soup = BeautifulSoup(response.text, "html.parser")

    # Save raw HTML for debugging
    with open("file.txt", "w", encoding="utf-8") as f:
        f.write(soup.prettify())

    pricing_data = []

    # Find all model blocks dynamically
    model_blocks = soup.find_all("li", class_=lambda x: x and "pricing" in x)

    for model_block in model_blocks:
        # Extract model name dynamically
        model_name_tag = model_block.find(["h3", "h2", "h1"])
        model_name = model_name_tag.get_text(strip=True) if model_name_tag else "Unknown Model"

        # Extract all pricing variations inside the model
        variations = model_block.find_all("li")  # Generic find to adapt to variations

        for variation in variations:
            cols = variation.find_all("div")

            if len(cols) >= 2:
                if "Llama" in model_name:
                    model_size = cols[0].get_text(strip=True).replace("Â\xa0", " ") if cols[0] else "Unknown Model Size"
                    model_type = cols[1].get_text(strip=True).replace("Â\xa0", " ") if cols[1] else "Unknown Model Type"

                    lite_price = clean_price(cols[2].get_text(strip=True)) if len(cols) > 2 else "Unknown Price"
                    turbo_price = clean_price(cols[3].get_text(strip=True)) if len(cols) > 3 else "Unknown Price"

                    pricing_data.append({
                        "Model Size": model_size,
                        "Model Type": model_type,
                        "Lite Price": lite_price,
                        "Turbo Price": turbo_price
                    })

                else:
                    variant_name = cols[0].get_text(strip=True).replace("Â\xa0", " ") if cols[0] else "Unknown Variant"
                    raw_price = cols[1].get_text(strip=True) if len(cols) > 1 else ""
                    price = clean_price(raw_price)  # Clean up price text

                    pricing_data.append({
                        "Model": model_name,
                        "Variant": variant_name,
                        "Price": price
                    })

    return pricing_data

# Fetch and print model pricing dynamically
pricing_info = get_all_model_pricing()
for entry in pricing_info:
    print(entry)

    df = pd.DataFrame(pricing_info)
    
    # Save to CSV
    df.to_csv("output.csv", index=False)