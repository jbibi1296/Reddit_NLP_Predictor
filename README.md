First, we started off with our [Problem Statement](#Problem-Statement)

# Problem Statement
We at T-mobile, would like to make sure that only posts that are meant for T-mobile are put up on our website. 
We have seen a bunch of posts that are meant for sprint popping up on our pages. 
To combat this issue we would like to create a model that can make sure that no verizon posts go up on our site.
In order to test our models, we decided to pull data from reddit.

# Methodology
We were approached with the task of building a prediction model. This model is meant to take in an post and determine whether it is meant for T-Mobile or not.
We at T-mobile really value our customers and we only want them to see posts that are meant for them to see and not spam from other websites.
We would like our model to be as accurate as possible in predicting whether a post belongs to T-mobile or not.<br>

We decided to use reddit posts to create our model. We designed a web-scraping tool to go and pull around 1000 posts from both the T-mobile subreddit and the Verizon subreddit.<br>

The posts were very messy and filled with data that we do not want such as random punctuation, white space, and special characters.<br>
We cleaned the posts and came across another issue, there were many words in the data that were either words that have no value to our model, or words that are essentially the same root such as (cook, cooking, cooks...).<br>
First, we removed all of these meaningless "stop words", then we also removed words that would be too obvious for our model such as "tmobile" or "Comcast". We used a device called a 'lemmatizer'. This device goes through each word and tries to bring it down to the root of each word.<br>
Once our posts were done being lemmatized, we were able to move to the next step.<br>

There are many different ways to prepare the data for these Natural Language Processing (NLP) models. <br>
Because of this, we decided to run a bunch of combinations of models and classifiers and create a grid with each combination and it's respective scores.<br>
These combinations were also run through a GridSearch function which checked for the best parameters for each model.<br>
Once we ran all of our models, we looked through them all to find whichever model best fit our problem.<br>

Once we chose the best fit model, we decided to make a function that lets you input text and tells you whether it is a part of the t-mobile subreddit or sprint.<br>
This is helpful as it would be a way for us to show off the accuracy of our model.