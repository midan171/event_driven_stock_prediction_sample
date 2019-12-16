# event_driven_stock_prediction_sample
A python application that uses web scraped news headlines of the day (from 'https://inshorts.com/en/read') to predict the probability of a stock having a higher closing price the next day.

This application uses a pre-trained FF Neural Network that uses advanced NLP techniques described in the work of X Ding, Y Zhang, T Liu and J Duan with title: "Deep learning for event-driven stock prediction" (2015), to preprocess some news headlines from financial news sites. The overall preprocessing of the news headlines takes a number of headlines and converts them into a 300 dimention vector, which represents the average Word Embedding of the day.

To run the programm:
Run market_simulator.py which uses 4 functions (2 in this version):

-get_todays_headlines(): Scraping the headlines

-extract_events(): Using reverd-latest.jar to extract events from the headlines

-events_to_wb(): Turn events into Word Embeddings (Not Included, commented out)

-get_prediction(): Get predictions of 5 Companies using the result of a pre-trained Network that uses the Word Embeddings of the Day as                      input. (Not included, commented out)

To get the predictions successfully you need a Word Embedding Dictionary that will be created by using skip-gram algorithm to learn the repressentation of words, to word Embeddings of 100 dimentions. I do not include this file in this repository. You can emai me if you need further information.





