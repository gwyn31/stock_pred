# -*- coding: utf-8 -*-
"""
Processing tweets data
"""

import os
import json

from pandas import Series


def load_senti_words_dict(fp: str) -> dict:
    """
    Load the sentiment dictionary
    :param fp: sentiment dictionary file path, which is "SentiWordNet_3.0.0"(removing the comment part)
    :return: {word: sentiment score}
    """
    senti_words_dict = dict()
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            item = line.strip().split("\t")
            words = (w[:-2].lower() for w in item[4].strip().split(" "))  # convert to lower then removing the "#1" or "#2"
            pos_score, neg_score = float(item[2]), float(item[3])  # get the positive and negative scores
            for w in words:
                if w not in senti_words_dict:
                    senti_words_dict[w] = (pos_score, neg_score)
                else:
                    continue
    return senti_words_dict


class TweetProcessor:

    """
    Class for processing tweet data
    """
    
    def __init__(self, senti_words_dict: dict):
        """
        :param senti_words_dict: sentiment words dictionary
        """
        self.senti_words_dict = senti_words_dict
    
    @staticmethod    
    def get_tweets_file_words_count(fp: str):
        """
        Given a tweet data file(corresponding to a certain date's all tweets about a stock),
        generate each tweet's words count
        :param fp: tweet file path
        """
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:  # each line is a tweet item
                tweet_item = json.loads(line)  # convert the line string to a dict
                tweet = tweet_item["text"]  # get the tweet(already parsed)
                words_count = dict()
                for w in tweet:
                    w = w.lower()
                    if w not in words_count:  # if this word never appear
                        words_count[w] = 1  # record its first appearance
                    else:
                        words_count[w] += 1  # else, frequency + 1
                yield words_count
    
    def calc_single_day_score(self, fp: str) -> float:
        """
        Calculate the sentiment score of a stock in a certain day
        :param fp: tweet file path
        :return: total sentiment score
        """
        score = 0
        for tweet_words_count in self.get_tweets_file_words_count(fp):  # traverse all tweets' word counts
            for w, num in tweet_words_count.items():  # traverse all words
                if w in self.senti_words_dict:  # if the word is a sentiment word
                    pos_score, neg_score = self.senti_words_dict[w]  # get its positive and negative score
                    score += num * (pos_score - neg_score)  # calculate the sentiment score
                else:
                    continue
        return score
    
    def get_stock_senti_history(self, tweets_dir: str) -> Series:
        """
        Calculate the sentiment scores history
        :param tweets_dir: directory that stores all history tweet data files of a ceratin stock
        :return: the sentiment score history
        """
        senti_history = Series(name="senti_score")
        for fname in os.listdir(tweets_dir):
            fp = os.path.join(tweets_dir, fname)
            score = self.calc_single_day_score(fp)
            senti_history[fname] = score
        return senti_history
    

def main():
    senti_dict_fp = "./data/senti_words/senti_words_clean.txt"
    history_data_dir = "./data/raw_price_train"
    all_tweets_dir = "./data/tweet_test"
    senti_words_dict = load_senti_words_dict(senti_dict_fp)
    tweets_processor = TweetProcessor(senti_words_dict)
    for dir_name in os.listdir(all_tweets_dir):
        print("Analyzing tweets from: {0}...".format(dir_name))
        tweets_dir = os.path.join(all_tweets_dir, dir_name)
        senti_history = tweets_processor.get_stock_senti_history(tweets_dir)
        stock_id = dir_name[0]
        save_fp = os.path.join(history_data_dir, "{0}_r_senti_score_test.csv".format(stock_id))
        senti_history.to_csv(save_fp, index=True, index_label="Date", header=True)  # 保存结果
        print("Senti history scores saved to: {0}".format(save_fp))


if __name__ == '__main__':
    main()
