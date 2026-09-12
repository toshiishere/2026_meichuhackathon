# Google Chrome includes H.264 playback; bundled open-source Chromium does not.
FROM mcr.microsoft.com/playwright:v1.56.1-noble
RUN npx --yes playwright@1.56.1 install chrome
WORKDIR /app
