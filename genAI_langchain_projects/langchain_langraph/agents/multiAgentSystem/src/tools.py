from ddgs import DDGS
from bs4 import BeautifulSoup
from langchain_classic.tools import tool
import requests

#
@tool
def web_search(query:str)->str:
    '''Search the web with given query for up to date information.Return titles ,urls ansd snippets. '''
    try :
        results = DDGS().text(query,max_results = 5)
    except Exception as e:
        return f"Search failed:{e}"
    # print(results)
    return " ".join([f"Title:{r['title']}\nUrl:{r['href']}\nSnippet:{r['body']}" for r in results])

@tool
def scrape_url(url:str):
    '''Fetch the important contetnt from given website url'''
    resposne = requests.get(url = url, headers={"User-Agent": "Mozilla/5.0"},timeout=10)
    soup = BeautifulSoup(markup=resposne.text, features="html.parser")

    for tag in soup(["script","style","nav","header","footer"]):
        tag.decompose()

    text = soup.get_text(separator="\n",strip=True)
    return text[:2000]

