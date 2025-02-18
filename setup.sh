#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# (Optional) If using Playwright with headless Chromium, install dependencies
playwright install-deps

#!/bin/bash
pip install --upgrade pip  # Upgrade pip to avoid old dependency issues
playwright install --with-deps  # Install Playwright with necessary dependencies
streamlit run scraping/scraper-ui.py  # Start the Streamlit app


echo "Setup completed successfully!"

