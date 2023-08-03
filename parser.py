import requests
from bs4 import BeautifulSoup
import time
from urllib.request import urlopen
import pandas as pd
pages = 40
reviews_dict = {'review_text' : [], 'is_spoiler' : []}
for page in range(pages):
    page_url = f'https://www.livelib.ru/reviews/~{page}'
    reviews_page = urlopen(page_url)
    html_bytes = reviews_page.read()
    reviews_page = html_bytes.decode('utf-8')
    page_soup = BeautifulSoup(reviews_page,  'html.parser')
    review_urls = page_soup.find_all('a', {'class': 'footer-card__link'})
    for url in review_urls:
        review_page = urlopen('https://www.livelib.ru' + url['href'])
        html_bytes = review_page.read()
        review_page = html_bytes.decode('utf-8')
        review_soup = BeautifulSoup(review_page,  'html.parser')
        spoiler = review_soup.find('span', {'class': 'lenta-card__spoiler'})
        if spoiler:
            reviews_dict['is_spoiler'].append(0)
        else:
            reviews_dict['is_spoiler'].append(1)
        review = review_soup.find('div', {'id': 'lenta-card__text-review-full'})
        review_text = ' '.join([par.get_text() for par in review.find_all('p')])
        reviews_dict['review_text'].append(review_text)
    time.sleep(7)
reviews = pd.DataFrame.from_dict(reviews_dict)
reviews.to_csv('reviews.csv')