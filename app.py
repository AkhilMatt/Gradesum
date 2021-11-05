#Standard
import time

#Third-party
from flask import Flask, render_template, request
from gensim.summarization import summarize 
#import spacy
from sumy.parsers.plaintext import PlaintextParser 
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer 

#Local
from essay_grading import grade
from nltk_summarization import nltk_summarizer
from spacy_summarization import text_summarizer 

#nlp = spacy.load('en_core_web_sm')
app = Flask(__name__)

# Sumy 
def sumy_summary(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,3)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result

# Reading Time
#def readingTime(mytext):
#	total_words = len([token.text for token in nlp(mytext)])
#	estimatedTime = total_words/200.0
#	return estimatedTime


@app.route('/')
def index():
        return render_template('compare_summary.html')

@app.route('/compare_summary')
def compare_summary():
        return render_template('compare_summary.html')

@app.route('/comparer',methods=['GET','POST'])
def comparer():
	#start = time.time()
	if request.method == 'POST':
		rawtext = request.form['rawtext']
		#final_reading_time = readingTime(rawtext)
		final_summary_spacy = text_summarizer(rawtext)
		#summary_reading_time = readingTime(final_summary_spacy)
		# Gensim Summarizer
		final_summary_gensim = summarize(rawtext)
		#summary_reading_time_gensim = readingTime(final_summary_gensim)
		# NLTK
		final_summary_nltk = nltk_summarizer(rawtext)
		#summary_reading_time_nltk = readingTime(final_summary_nltk)
		# Sumy
		final_summary_sumy = sumy_summary(rawtext)
		#summary_reading_time_sumy = readingTime(final_summary_sumy) 
		#end = time.time()
		#final_time = end-start
	return render_template('compare_summary.html',ctext=rawtext,final_summary_spacy = final_summary_spacy, final_summary_gensim=final_summary_gensim,final_summary_nltk=final_summary_nltk, final_summary_sumy = final_summary_sumy)
# final_time = round(final_time,2),final_reading_time = round(final_reading_time,2), summary_reading_time=round(summary_reading_time,2),summary_reading_time_gensim=round(summary_reading_time_gensim,2)
#, summary_reading_time_sumy=round(summary_reading_time_sumy,2),
#                               summary_reading_time_nltk=round(summary_reading_time_nltk,2)

@app.route('/essay_grading')
def essay_grading():
        return render_template('essay_grading.html')

@app.route('/essay_grader',methods=['GET','POST'])
def grader():
        #start = time.time()
        if request.method == 'POST':
                essay = request.form['essay']
                #reading_time = readingTime(essay)
                predicted_score = grade(essay)
                #end = time.time()
                #final_time = end-start
        return render_template('essay_grading.html', ctext=essay, predicted_score=predicted_score)
# final_reading_time = round(reading_time,2), final_time = round(final_time,2)

if __name__ == '__main__':
    app.run(debug = True)
