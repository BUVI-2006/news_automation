import os 
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timezone, timedelta
import requests
import json 
import time 
import hashlib
from dotenv import load_dotenv

load_dotenv()


api_key = os.environ.get('MARKET_API_KEY')

def init_firebase():
   
    firebase_key_raw = os.environ.get('FIREBASE_KEY')
    if not firebase_key_raw:
        raise ValueError("FIREBASE_KEY environment variable not set")
        
    firebase_key = json.loads(firebase_key_raw)
    cred = credentials.Certificate(firebase_key)
    
   
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    
    return firestore.client()

def news_store(db):
    tickers = ['ABCB', 'ACNB', 'ALRS', 'AMAL', 'AMTB']
    url = 'https://www.alphavantage.co/query'

    now = datetime.now(timezone.utc)
    today_10am = now.replace(hour=10, minute=0, second=0, microsecond=0)

    if now.hour < 10:
        today_10am -= timedelta(days=1)

    yesterday_10am = today_10am - timedelta(days=1)
    time_from = yesterday_10am.strftime('%Y%m%dT%H%M')
    time_to = today_10am.strftime('%Y%m%dT%H%M')

    for ticker in tickers:
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker, 
            "time_from": time_from,
            "time_to": time_to,
            "limit": 1000,
            "apikey": api_key
        }

        try:
            response = requests.get(url, params=params)
            data = response.json()
            print(f"\n{ticker} API Response:", json.dumps(data, indent=2))

            
            if 'feed' not in data:
                print(f"No feed found for {ticker}. API Response: {data.get('Note', 'Unknown Error')}")
                continue

            articles = data['feed']

            for article in articles:
                dt = datetime.strptime(article['time_published'], "%Y%m%dT%H%M%S")
                dt = dt.replace(tzinfo=timezone.utc)

                clean_data = {
                    "title": article['title'],
                    "description": article.get("summary", ""),
                    "published_date": dt.isoformat().replace("+00:00", "Z"),
                    "url": article.get("url", ""),
                    "sentiment_score": article.get("overall_sentiment_score")
                }

               
                doc_id = hashlib.md5(article["url"].encode()).hexdigest()

                
                db.collection('news').document(ticker).collection("articles").document(doc_id).set(clean_data)
            
            print(f"Successfully processed {ticker}")
            
         
            time.sleep(60) 

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

if __name__ == "__main__":
    db = init_firebase()
    news_store(db)