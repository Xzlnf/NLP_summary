import spacy
import pandas as pd
import nltk
import re
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from keybert import KeyBERT
import json

#Summmarizing

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

summarizor = pipeline('summarization', model = 'facebook/bart-large-cnn')

#keyword extractor

kw_model = KeyBERT()
# key_words = kw_model.extract_keywords(details,keyphrase_ngram_range=(2,2), top_n=3)
# key_words[0][0]

#SMS_summarizor

class SMS_summarizor(object):
    
    def __init__(self,paragraph):
        self.para = paragraph
        #preprocessing the data
        self.__para1 = self.__preprocessor(self.para)
        
    def __preprocessor(self,text):
        #Extend the preprocessor function as for the requirements
        text = re.sub(r'\n',"",text)
        return text
    
    def summarize_sms(self, max_length,min_length):
        """
        Summarize the given content according the max_length and min_length parameters

        max_length ----> int
        min_length ----> int
        
        """
        #asserting to remove errors and warnings
        assert max_length < len(tokenizer.encode(self.__para1)), "Max length must be smaller than {} ".format(len(tokenizer.encode(self.__para1)))
        assert len(tokenizer.encode(self.__para1)) < tokenizer.model_max_length, "Text document must contain more less information"
        #summarizing the content
        summary = summarizor(self.__para1,max_length=max_length, min_length=min_length)
        self.text = summary[0]['summary_text']
        
    def title_generator(self,number_of_words):
        """
        Generating the tiitle usinng key_word extraction
        number_of_words ---> Define the title of the summary
                              type--->int
        """
        assert type(number_of_words) == int,"number of words must be an integer"
        assert number_of_words <3, "Limit the words to 3"
        key_words = kw_model.extract_keywords(self.__para1,keyphrase_ngram_range=(number_of_words,number_of_words), top_n=3)
        self.title = key_words[0][0]
        
    def save_json(self,path,json_file_name):
        """
        Save the Json file of the summarized content and it's generated title.
        path ---> Path must be ended with \
                    ex-:C:\Lachin\DataSets\SMS\
        json_file_name ---> name.json
        """
  
        assert type(path) == str and type(json_file_name) == str , "Pathn and json_file_name must be string"
        assert json_file_name.endswith("json"),"file_name must be a .json extension"
        dict1 = {
            "SMS_Summarizor":{
                "Title": str(self.title),
                "Summary": str(self.text)
            }
        }
        
        location = open(path + json_file_name,'w')
        json.dump(dict1,location, indent = 2)
        print("Json File successfull saved.")

