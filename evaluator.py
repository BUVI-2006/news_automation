from datetime import datetime 
import pandas as pd  
import yfinance as yf 
import os 
import nltk 
from datetime import datetime ,timedelta
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from fastapi import FastAPI
import firebase_admin
from firebase_admin import firestore,credentials 
from dotenv import load_dotenv
from pydantic import BaseModel
import uvicorn
import pickle 
from huggingface_hub import InferenceClient
from statsmodels.tsa.statespace.sarimax import SARIMAX




load_dotenv()


app=FastAPI()

cred=credentials.Certificate('serviceaccountkey.json')
firebase_admin.initialize_app(cred)

HF_TOKEN=os.environ.get('HF_TOKEN')

   
client=InferenceClient(token=HF_TOKEN)



@app.get('/')
def index():
   
   return {
      'Instruction':"end point returns risk scores"
   }


def lexrank_summary(text,sentence_count=1):
    if not text or len(text.strip())==0:
        return ""
    
    parser=PlaintextParser.from_string(text,Tokenizer("english"))

    summarizer=LexRankSummarizer()

    summary=summarizer(parser.document,sentence_count)

    return "".join(str(sentence) for sentence in summary)


def liquidity_threshold(liquidtiy_value):

   SAFE=1.0
   WARNING=0.8
   CRITICAL=0.6

   if liquidtiy_value>=SAFE:
      liquidity_score=10

   elif liquidtiy_value>=WARNING:
        liquidity_score=40
    
   elif liquidtiy_value>=CRITICAL:
        liquidity_score=70

   else:
        liquidity_score=95

   return liquidity_score


def drawdown_threshold(drawdown_value):
  SAFE = -0.05        
  WARNING = -0.15     
  CRITICAL = -0.25   
    
  if drawdown_value >= SAFE:  
        drawdown_score = 10
  elif drawdown_value >= WARNING:
        drawdown_score = 40
  elif drawdown_value >= CRITICAL:
        drawdown_score = 70
  else:
        drawdown_score = 95  
    
  return drawdown_score


def stress_threshold(stress_value):
   SAFE=0.2
   WARNING=0.5
   CRITICAL=0.8

   if stress_value<=SAFE:
       stress_score=10
   
   elif stress_value<=WARNING:
       stress_score=40

   elif stress_value<=CRITICAL:
       stress_score=70

   else:
       stress_score=95

   return stress_score





def data_computer(stock,db):
    """Final working code for RISK prediction"""

#============================Stock evaluation based on dates ====================================
    articles_ref=db.collection('news').document(stock).collection("articles")
    docs=articles_ref.stream()


    min_date=None
    max_date=None

    if not docs :
       return None 
    

    for doc in docs :
        data=doc.to_dict()


        if "publish_date" not in data:
            continue 

        t = pd.to_datetime(data['publish_date'], format='ISO8601').date()

    
        if max_date is None or t>max_date:
          max_date=t

        if min_date is None or t<min_date:
           min_date=t

        

    
    end =max_date.strftime("%Y-%m-%d")
    start=min_date.strftime("%Y-%m-%d")




    df=yf.download(stock,start=start,end=end)


    df.columns=df.columns.droplevel(1)

    
    df.reset_index(inplace=True)
    
    df=df.sort_values('Date')
    df["returns%"] = (df['Close']-df['Open'])*100/df['Open']                                           # returns 
    df["dollar_volume"]=(df['Close']*(df["Volume"].astype('float64')))                                #liquidity
    df["Stress"]=((df['High']-df['Low'])/df['Close'])                                                #stress value                                              
    df["Trend"]=((df['Close'].rolling(5).mean()-df['Close'].rolling(20).mean()))                     #trend (uptrend or downtrend)
    df["Volume_spike"]=(df['Volume']/(df['Volume'].rolling(10).mean()))                              #volumespike 
    df["Gap"]=((df['Open']-(df['Close'].shift(1)))/df['Close'])                                    #Gap(overnight reaction to news)
    df["Drawdown"]=(df['Close']-(df['Close'].rolling(30).max()))/(df['Close'].rolling(30).max())  # Loss of confidence(Drawdown)
    df["Momentum"]=(df["Close"]-df["Close"].shift(10))
    df["liquidity_shock"]=df["dollar_volume"]/(df["dollar_volume"]).rolling(20).mean()

    df=df.bfill()

  
#===============================Lexrank summarizer =====================================

    articles_ref=db.collection('news').document(stock).collection("articles")

    docs=articles_ref.stream()


    sid=SentimentIntensityAnalyzer()

    rows=[]

    for doc in docs :
       
       doc=doc.to_dict()
       
       if "publish_date" not in doc:
          continue

       text=doc['title']+'.'+doc['description']
       lexrank=lexrank_summary(text)
       sentiment=(0.0 if pd.isna(lexrank) or lexrank.strip()=="" else sid.polarity_scores(lexrank)['compound'])
       rows.append({'Date':doc['publish_date'],'Lexrank_summary':lexrank,'Sentiment':sentiment})


    
    sentiment_data=pd.DataFrame(rows)



# ======================merging the final data for prediction=================================

    df['Date']=pd.to_datetime(df['Date'],utc=True).dt.date
    sentiment_data['Date']=pd.to_datetime(sentiment_data['Date'],format='ISO8601',utc=True).dt.date
    merge_data=df.merge(sentiment_data[['Date','Sentiment']],on='Date',how='left')
    merge_data['last_news_date']=merge_data['Date'].where(merge_data['Sentiment'].notna())
    merge_data['last_news_date']=merge_data['last_news_date'].ffill()
    merge_data['last_news_date']=pd.to_datetime(merge_data['last_news_date'])  
    merge_data['Date']=pd.to_datetime(merge_data['Date'])
    merge_data['days_since_news']=(merge_data['Date']-merge_data['last_news_date']).apply(lambda x: x.days)    

# =====================measuring the impact of news=======================================
    max_days=5
    merge_data['Sentiment']=merge_data['Sentiment'].ffill()
    merge_data['Sentiment'] = merge_data.apply( lambda row: 0 if row['days_since_news'] > max_days else row['Sentiment'], axis=1 ) 

# ===================calculating the lags==========================
    merge_data=merge_data.drop(columns=[c for c in merge_data.columns if c.endswith('_lag1')])

    print("After merging data:",merge_data.shape)

    cols=['Trend','Volume_spike','Gap','Sentiment','days_since_news','Drawdown','Stress']


    for col in cols: 
        merge_data[f'{col}_lag1']=merge_data[col].shift(1)


    merge_data=merge_data.dropna()

    return merge_data
      

class PredictRequest(BaseModel):
   stock:str



# Just import the time series model and evaluate prediction under this ....
@app.post("/predict")      # from post , only stock is given ==> 
def forecast_series(data:PredictRequest):
   data=data.model_dump()
   stock=data['stock']

   db=firestore.client()

   merge_data=data_computer(stock=stock,db=db)

   with open(f'{stock}_corporation/liquidity_model.pkl','rb') as liquidity:
      liquidity_model=pickle.load(liquidity)

   with open(f'{stock}_corporation/stress_model.pkl','rb') as stress:
      stress_model=pickle.load(stress)
    
   with open(f'{stock}_corporation/drawdown_model.pkl','rb') as drawdown:
      drawdown_model=pickle.load(drawdown)

    
   liquidity_forecast=liquidity_model.forecast(steps=len(merge_data),exog=merge_data[['Trend_lag1', 'Volume_spike_lag1', 'Gap_lag1','Sentiment_lag1', 'days_since_news_lag1']])
   stress_forecast=stress_model.forecast(steps=len(merge_data),exog=merge_data[['Volume_spike_lag1', 'Gap_lag1','Sentiment_lag1','Drawdown_lag1','Stress_lag1']])
   drawdown_forecast=drawdown_model.forecast(steps=len(merge_data),exog=merge_data[['Trend_lag1','Volume_spike_lag1', 'Gap_lag1','Sentiment_lag1', 'days_since_news_lag1','Drawdown_lag1']])

   x_values=merge_data['Date']

   last_liq_score=liquidity_forecast.tolist()[-1]
   last_draw_score=drawdown_forecast.tolist()[-1]
   last_stress_score=stress_forecast.tolist()[-1]

   liquidity_score=liquidity_threshold(last_liq_score)
   drawdown_score=drawdown_threshold(last_draw_score)
   stress_score=stress_threshold(last_stress_score)

   # According to bank rule for risk calcualtion in BASEL III

   risk_score=(liquidity_score * 0.40 + 
                  stress_score * 0.35 + 
                  drawdown_score * 0.25)
   
   formatted_drawdown = f"{last_draw_score:.1%}"  # Results in %
   formatted_stress = f"{last_stress_score:.1%}"
   
   messages = [
    {
        "role": "system", 
        "content": """You are a Senior Bank Risk & Compliance Officer. 
        Your task is to analyze portfolio metrics against Basel III and internal liquidity standards.
        
        CRITICAL INSTRUCTIONS:
        - Return the analysis in a clean, structured format.
        - Use specific financial terminology (e.g., LCR, NSFR, Value-at-Risk).
        - If any score is critical, prefix the analysis with '!!! CRITICAL ALERT !!!'.
        - Ensure the output is concise enough for a dashboard widget."""
    },
    {
        "role": "user", 
        "content": f"""
        Analyze these live portfolio metrics:
        - Liquidity Ratio: {last_liq_score} (Internal Benchmark: >1.0)
        - Max Drawdown: {formatted_drawdown}% (Limit: 15%)
        - Stress Test Score: {formatted_stress} (Threshold: 70)
        - Combined Risk Score: {risk_score}

        Output your response exactly in this structure:
        ### 1. Executive Summary
        [One sentence on overall health]

        ### 2. Metric Breakdown
        - **Liquidity:** [Analysis]
        - **Drawdown:** [Analysis]
        - **Stress:** [Analysis]

        ### 3. Compliance & Action
        - **Status:** [Compliant/Non-Compliant]
        - **Priority:** [Low/Medium/High/Critical]
        - **Immediate Action:** [Step 1, Step 2]
        """
    }
   ]


   response=client.chat_completion(
       model="meta-llama/Llama-3.2-3B-Instruct",
       messages=messages,
       max_tokens=500,
       temperature=0.0
   )
   
   
   


   return {
      "liquidity":{
         "x":x_values,
         "y": liquidity_forecast.tolist()
      },
      "stress":{
         "x":x_values,
         "y":stress_forecast.tolist()
      },
      "drawdown":{
         "x":x_values,
         "y":drawdown_forecast.tolist()
      } ,
      "risk_score":risk_score,

      "LLM_response":response.choices[0].message.content
      
      }








if __name__=="__main__":
   uvicorn.run(app,host="127.0.0.1",reload=True)


   
   