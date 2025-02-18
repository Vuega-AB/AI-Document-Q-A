#!/bin/bash

# Update package lists
apt-get update

# Install missing dependencies
apt-get install -y \
  libgtk-4-dev \
  libgraphene-1.0-0 \
  libgstreamer-gl1.0-0 \
  libgstreamer-plugins-bad1.0-0 \
  libavif15 \
  libenchant-2-2 \
  libsecret-1-0 \
  libmanette-0.2-0 \
  libgles2-mesa

# Install Playwright browsers
playwright install
