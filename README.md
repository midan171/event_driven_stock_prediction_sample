# event_driven_stock_prediction_sample
A python application that uses web scraped news headlines of the day (from 'https://inshorts.com/en/read') to predict the probability of a stock having a higher closing price the next day.

This application uses a pre-trained FF Neural Network that uses advanced NLP techniques described in the work of X Ding, Y Zhang, T Liu and J Duan with title: "Deep learning for event-driven stock prediction" (2015), to preprocess some news headlines from financial news sites. The overall preprocessing of the news headlines takes a number of headlines and converts them into a 300 dimention vector, which represents the average Word Embedding of the day.


To run the programm:
-download the reverb-latest.jar file from http://reverb.cs.washington.edu/README.html and place it inside the working directory. There      is an example of how the finall folder should look in the screenshots directory. 

To get the predictions successfully you need a Word Embedding Dictionary that will be created by using skip-gram algorithm to learn the repressentation of words, to word Embeddings of 100 dimentions.
Retrain the Word Embedding dictionary by running "make_wb_dict.py" after un-commenting the funtion "list_files_train_model()" and putting a large number of news files in the "news_files" directory so that the programm can be trained on this data and make a decent WB dictionary. After the training is done, comment out the function again.

Then Run market_simulator.py which uses 4 functions (2 in this version):

-get_todays_headlines(): Scraping the headlines

-extract_events(): Using reverd-latest.jar to extract events from the headlines

-events_to_wb(): Turn events into Word Embeddings

-get_prediction(): Get predictions of 5 Companies using the result of a pre-trained Network that uses the Word Embeddings of the Day as                      input.

Among other puthon dependencies, this programm requires that you can run ".jar" files in your system, so that it can call the "reverb-latest.jar" tool which performs the event extraction process.

*Some paths in the files may be broken especially if you are not using windows operating system, but they can be easilly fixed.





