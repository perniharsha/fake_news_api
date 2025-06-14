from newspaper import Article

def extract_article(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.title, article.text
    except Exception as e:
        print(f"failed to extract article: {e}")
        return None, None