#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install

# (Optional) If using Playwright with headless Chromium, install dependencies
playwright install-deps

echo "Setup completed successfully!"
