import os
import subprocess
from yahoo_fin import stock_info as si
from datetime import date
from time import sleep
import requests
from bs4 import BeautifulSoup

import numpy as np
import pandas as pd
from make_wb_dict import clean_text
from make_wb_dict import load_model
import make_wb_dict as WB
from keras.models import load_model


#get today's date
today = date.today()
#today = "2019-11-01"
print("Today's date:", today)

#path
dir_path = os.path.dirname(os.path.abspath(__file__))
today_file = os.path.join(dir_path,'today\\'+str(today))
today_file_num = os.path.join(dir_path,'today\\'+str(today)+'_num')

today_events_file = os.path.join(dir_path,'today\\'+str(today)+'_events.csv')
today_wb_file = os.path.join(dir_path,'today\\'+str(today)+'_wb.csv')

def get_todays_nums():
    #get yahoo data
    num_data = si.get_quote_table('^GSPC')
    f = open(today_file_num, "w", encoding="utf-8")
    f.write(str(num_data['Previous Close'])+',')
    f.write(str(num_data['Open']))
    print('Previous Close: ',num_data['Previous Close'])
    print('Open: ',num_data['Open'])
    f.close()



def get_todays_headlines():
    #request,response
    url = 'https://inshorts.com/en/read'
    response = requests.get(url)
    
    #write file
    soup = BeautifulSoup(response.text, 'lxml')
    headlines = soup.find_all(attrs={"itemprop": "headline"})
    print(len(headlines))
    f = open(today_file, "w", encoding="utf-8")
    for headline in headlines:
        print(headline.text)
        f.write(headline.text+'.\n')
    f.close()



def extract_events():
    #cmnd = command to run reverb-latest.jar
    com="java -Xmx512m -jar "
    reverb_path = os.path.join(dir_path,"reverb-latest.jar ")
    cmnd = com + reverb_path
    
    comnd = cmnd+today_file
    print(comnd)
    proc = subprocess.check_output(comnd)
    try:
        file = open(today_events_file, "w", encoding="utf-8")
    
        file.write(proc.decode("utf-8",errors='ignore'))
    
        file.close()
    except IOError:
        print("IO Error")
    except TypeError:
        print("Type Error")
    except UnicodeDecodeError:
        print("UnicodeDecodeError")

def events_to_wb():
    EMBEDDING_DIM = 100

    #in out paths
    model_path = os.path.join('C:\\Users\\midan\\Desktop\\ReVerb\\Event_extractor_py','wb_models\\_model.word2vec')
    
    #load wb model
    model = WB.load_model(model_path)

    #open csv
    df = pd.read_csv(today_events_file,sep='\t')
    df = df.dropna()
    

    #filter out low rate extractions
    indexNames = df[df.iloc[:,11] < 0.7 ].index
    df.drop(indexNames , inplace=True)
    print("Confident events number: ",df.shape[0])

    #declair new df
    WB1 = np.zeros(EMBEDDING_DIM)
    WB2 = np.zeros(EMBEDDING_DIM)
    WB3 = np.zeros(EMBEDDING_DIM)
    new_df_col = ['date'] + list(WB1) + list(WB2) + list(WB3)
    new_df = pd.DataFrame(columns=new_df_col)


    #init variables
    O1=[]
    P=[]
    O2=[]
    dates=[]
    rows=[]
    avg= np.zeros(100)
    avg_value = avg.tolist()
    indexed_tokens = 0
    not_indexed_tokens = 0

    for i in range(0,len(df)):
        try:
            arg1=df.iloc[i,2]#______2,3,4 or 15,16,17 norm event args
            arg2=df.iloc[i,3]
            arg3=df.iloc[i,4]
            date=today
            print(arg1)
            print(arg2)
            print(arg3)
            if arg1 is not None and arg2 is not None and arg3 is not None:
                #get date
                date=today
                #tokenize
                o1 = clean_text(str(arg1)).split()

                o2 = clean_text(str(arg2)).split()

                o3 = clean_text(str(arg3)).split()
                #tokenize
                #o1 = str(arg1).split()
                #print("o1: ",o1)
                #o2 = str(arg2).split()
                #print("o2: ",o2)
                #o3 = str(arg3).split()
                #print("o3: ",o3)
                arg1_vecs=[]
                arg2_vecs=[]
                arg3_vecs=[]
                print(len(df))
                for j in range(0,len(o1),1):
                    try:
                        arg1_vecs.append(model.wv[o1[j]])
                        indexed_tokens = indexed_tokens + 1
                    except KeyError:
                        arg1_vecs.append(avg_value)
                        not_indexed_tokens = not_indexed_tokens + 1
                for j in range(0,len(o2),1):
                    try:
                        arg2_vecs.append(model.wv[o2[j]])
                        indexed_tokens = indexed_tokens + 1
                    except KeyError:
                        arg2_vecs.append(avg_value)
                        not_indexed_tokens = not_indexed_tokens + 1
                for j in range(0,len(o3),1):
                    try:
                        arg3_vecs.append(model.wv[o3[j]])
                        indexed_tokens = indexed_tokens + 1
                    except KeyError:
                        arg3_vecs.append(avg_value)
                        not_indexed_tokens = not_indexed_tokens + 1
                
                
                dates.append(date)
                
                avg_arg1_vec = np.mean(arg1_vecs, axis=0).tolist()
                O1.append(avg_arg1_vec)

        
                avg_arg2_vec = np.mean(arg2_vecs, axis=0).tolist()
                P.append(avg_arg2_vec)

                avg_arg3_vec = np.mean(arg3_vecs, axis=0).tolist()
                O2.append(avg_arg3_vec)

                try:
                    row = [date] + list(avg_arg1_vec) + list(avg_arg2_vec) + list(avg_arg3_vec)
                    new_df.loc[i] = row
                except TypeError:
                    print("type error------------------------------------------------------------------")

                if (i%200 == 0):
                    print('did ',i,' out of ',len(df))
        except IndexError:
            print("Index error-------------------------------------------------------------------------")
            break



    print("---------------------------------")
    print("Indexed tokens= ", indexed_tokens)
    print("Not indexed tokens= ",not_indexed_tokens)

    new_df = new_df.groupby('date').mean()



    #create csv 
    new_df.to_csv(today_wb_file, sep='\t')
    

def get_prediction():       
    #open csv
    df = pd.read_csv(today_wb_file,sep='\t')
    print(df)
    new_df = df.iloc[:,1:301]
    #load model
    model_path = os.path.join(dir_path,'pred_models\\sp500_bloom_best.h5')
    model = load_model(model_path)
    prediction = model.predict(new_df)
    print('SP500: ',prediction)
    model_path = os.path.join(dir_path,'pred_models\googl_ep4001_ac_0.58.h5')
    model = load_model(model_path)
    prediction = model.predict(new_df)
    print('GOOGL: ',prediction)
    model_path = os.path.join(dir_path,'pred_models\\nke_ep4001_ac_0.59.h5')
    model = load_model(model_path)
    prediction = model.predict(new_df)
    print('NKE: ',prediction)
    model_path = os.path.join(dir_path,'pred_models\\wmt_ep4001_ac_0.53.h5')
    model = load_model(model_path)
    prediction = model.predict(new_df)
    print('WMT: ',prediction)
    model_path = os.path.join(dir_path,'pred_models\\qcom_ep4001_ac_0.56.h5')
    model = load_model(model_path)
    prediction = model.predict(new_df)
    print('QCOM: ',prediction)
    model_path = os.path.join(dir_path,'pred_models\\ibm_ep4001_ac_0.54.h5')
    model = load_model(model_path)
    prediction = model.predict(new_df)
    print('IBM: ',prediction)


#get_todays_nums()

get_todays_headlines()
extract_events()
events_to_wb()
get_prediction()



ok = input("Press Enter key to continue")


