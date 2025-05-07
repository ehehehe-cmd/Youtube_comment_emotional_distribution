import googleapiclient.discovery
import pandas as pd
from typing import Counter
import matplotlib.pyplot as plt
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = input("Please enter your api key") 

youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey = DEVELOPER_KEY)

request = youtube.commentThreads().list(
    part = "snippet",
    videoId = input("Please enter video id"),
    maxResults = 1000
)
response = request.execute()

comments = []

for item in response["items"]:
    comment = item["snippet"]["topLevelComment"]["snippet"]
    comments.append([
        comment["authorDisplayName"],
        comment["publishedAt"],
        comment["updatedAt"],
        comment["likeCount"],
        comment["textDisplay"]
        ])
df = pd.DataFrame(comments, columns=["author", "publishedAt", "updatedAt", "likeCount", "text"])



analyzer = SentimentIntensityAnalyzer()

result_data =[]

for item in df["text"]:
  scores = analyzer.polarity_scores(item)
  result_data.append([item, scores["pos"], scores["neg"], scores["neu"], scores["compound"]])

result_df = pd.DataFrame(result_data, columns=["text", "pos", "neg", "neu", "compound"])


visual_data = []
i = 0
for item in result_df["compound"]:
  if item >= 0.05:
    visual_data.append([i,"Possitive"])
  elif item <= -0.05:
    visual_data.append([i,"Negative"])
  else:
    visual_data.append([i,"Notr"])
  i += 1

final_df = pd.DataFrame(visual_data, columns=["index", "sentiment"])



final = Counter(final_df["sentiment"])

Labels = list(final.keys())
Values = list(final.values())

plt.pie(Values, labels=Labels, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('Emotional Distribution')
plt.show()