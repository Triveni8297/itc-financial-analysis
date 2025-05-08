# scrape.py
from crewai_tools import ScrapeWebsiteTool

def main(website_url):
    # To scrape only this site
    tool = ScrapeWebsiteTool(website_url)
    text = tool.run()

    # save text data in a local file
    with open('itcportal.txt', 'w', encoding='utf-8', errors='ignore') as f:
        f.write(text)
    print("✅ Saved scraped text to itcportal.txt")



if __name__ == "__main__":
    website_url = 'https://www.itcportal.com/'
    main(website_url)

