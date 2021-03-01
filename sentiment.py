#!/usr/bin/env python
# coding: utf-8

# This is a dataset of over a million headlines taken from the Australian news source ABC (Start Date: 2003-02-19 ; End Date: 2020-12-31).
# 
# - Calculate the sentiment score for every headline in the data. You can do this using the spaCyTextBlob approach that we covered in class or any other dictionary-based approach in Python.
# - Create and save a plot of sentiment over time with a 1-week rolling average
# - Create and save a plot of sentiment over time with a 1-month rolling average
# - Make sure that you have clear values on the x-axis and that you include the following: a plot title; labels for the x and y axes; and a legend for the plot
# - Write a short summary (no more than a paragraph) describing what the two plots show. You should mention the following points: 1) What (if any) are the general trends? 2) What (if any) inferences might you draw from them?
# 
# 
# - HINT: You'll probably want to calculate an average score for each day first, before calculating the rolling averages for weeks and months.
# 
# 
# __General instructions__
# 
# - For this assignment, you should upload a standalone .py script which can be executed from the command line or a Jupyter Notebook
# - Save your script as sentiment.py or sentiment.ipynb
# - Make sure to include a requirements.txt file and details about where to find the data
# - You can either upload the scripts here or push to GitHub and include a link - or both!
# - Your code should be clearly documented in a way that allows others to easily follow the structure of your script and to use them from the command line
# 
# 
# __Purpose__
# 
# This assignment is designed to test that you have a understanding of:
# 
# - how to perform dictionary-based sentiment analysis in Python;
# - how to effectively use pandas and spaCy in a simple NLP workflow;
# - how to present results visually, working with datetime formats to show trends over time

# In[1]:


import os #For integrating with operating systems
import spacy #For performing nlp-tasks
import pandas as pd #For creating dataframes
import matplotlib.pyplot as plt #For creating plots
from spacytextblob.spacytextblob import SpacyTextBlob #For sentiment analysis.

#Initialize spaCy
nlp = spacy.load("en_core_web_sm")
#Initialize spaCy text blob and add is a component to the spaCy nlp-pipeline
spacy_text_blob = SpacyTextBlob()
nlp.add_pipe(spacy_text_blob)


# In[2]:


in_file = os.path.join("..", "data", "headlines", "abcnews-date-text.csv") #Specify path to data.


# In[3]:


headlines = pd.read_csv(in_file) #Read the .csv-file as a dataframe called 'headlines'.


# In[4]:


headlines.sample(5) #Print 5 values from the 'headlines'-dataframe (sanity check)


# __Calculate sentiment__

# In[5]:


headlines['publish_date'] = pd.to_datetime(headlines.publish_date, format="%Y%m%d") #Converting the publish_date to a datetime format. Following the output above I specify that the dates are arranged as years, months, days (%Y%m%d). Y is upper case since the year is specified with century (i.e. four numbers).  
headlines = headlines.sort_values("publish_date") #Sorting the dates in chronological order.


# In[6]:


len(headlines) #Printing the lenght of the dataframe (i.e. the amount of comments in the dataset).


# In[7]:


headlines = headlines.sample(100000) #To reduce processing time, I take a sample of 100.000 comments. If you wish to process the full data just remove this line of code.


# In[8]:


#%%time #Specify that I want to print the time it takes me to run the cell.

sentiment_scores = [] #Define empty list for sentiment-scores.

for doc in nlp.pipe(headlines["headline_text"], batch_size=500): #For each headline in the headline_text column (iterated chronologically)...
    score = doc._.sentiment.polarity #... calculate text polarity and save it as 'score'...
    sentiment_scores.append(score) #... append 'score' to the 'sentiment_scores'-list


# In[9]:


len(sentiment_scores) #Ensuring that the code worked (sanity check).


# In[10]:


#Append list with the sentiment score into pandas dataframe
headlines["sentiment_score"] = sentiment_scores


# In[11]:


headlines.sample(5) #(sanity check)


# In[12]:


#Compute mean sentiment score for week and month
mean_week = headlines.resample("w",on ="publish_date").mean()
mean_month = headlines.resample("m",on ="publish_date").mean()


# In[13]:


print(mean_week) #Sanity check
print(mean_month) #Sanity check


# __As we can see from the above output, `mean_week` contains weekly averages of sentiment scores whereas `mean_month` contains monthly averages.__

# In[14]:


#Make plot of the weekly rolling mean of sentiment scores
week_plot = mean_week.plot(
    ylabel = "Sentiment Score", #Define the label for the y-axis.
    xlabel = "Date", #Define the label for the x-axis.
    ylim = (-0.2,0.2), #Set the window size on the y-axis.
    title = "Weekly rolling mean of sentiment scores") #Plot title


# In[15]:


#Save week_plot.
mean_week_plot = week_plot.get_figure()
mean_week_plot.savefig(os.path.join("..", "data", "mean_week_plot"))


# In[16]:


#Make plot of the monthly rolling mean of sentiment scores
month_plot = mean_month.plot(
    ylabel = "Sentiment Score",
    xlabel = "Date",
     ylim = (-0.2,0.2),
    title = "Monthly rolling mean of sentiment scores")


# In[17]:


mean_month_plot = month_plot.get_figure()
mean_month_plot.savefig(os.path.join("..", "data", "mean_month_plot"))


# __Plot inferences__
# 
# The two plots show that the news data is generally positive (Sentiment score > 0) in spite of the fluctuations on a weekly/monthly basis. We see that the variance in sentiment scores is larger on a weekly basis than on a monthly basis, indicating that the variance seen from week to week is quite similar in both directions and thus, to a certain extent, cancels each other out when analysed on a monthly basis.

headlines.to_csv(os.path.join("..", "data", "sentiment.csv")) #Write the dataframe as a .csv-file called 'file1'. #Write the dataframe as a .csv-file called 'sentiment.csv'.
