# Gradesum  
### Project description:
- The entire purpose of text summarization is to condense the source data into a concise version of text while preserving information content and overall meaning
- Grading essays takes up a significant amount of an instructor's valuable time, and hence is an expensive process. Automated Essay grading system allows us to assign scores to essays using computer programs

### Problem Statement:
To build an automated Text summarizer & Essay grading model

### Dataset:
Link: https://www.kaggle.com/c/asap-aes/data  
It consists of 12976 rows, 28 columns of which 6 columns are considered.

### Data description:
- essay_id : Unique id for each essay
- essay_set : A particular set number to which the eassay belongs (there are 8 distinct essay sets)
- essay : Consists of essays
- rater1_domain1 : Score given manually by an examiner 1
- rater2_domain2 : Score given manually by an examiner 2
- domain1_score : Average score of rater1_domain1 and rater2_domain2

### Tools used:
spaCy, Gensim, NLTK, Lexrank, Sumy, sklearn, Flask, MaterializeCSS

### Work Flow:
1. Essay Grader
    - Scaling grades from different sets
    - Pre-processing
    - Feature extraction using sklearn and nltk
    - Prediction using Support vector regressor
2. Text Summarizer using Sumy Lexrank and Gensim
    - Convert Paragraphs to sentences
    - Text Pre-processing
    - Find vector representation for every sentence
    - Construct similarity matrix between vectors
    - Similarity matrix is then converted into a graph
    - The top-ranked sentences form the final summary
3. Text Summarizer using Spacy and NLTK:
    - Convert Paragraphs to Sentences
    - Text Pre-processing
    - Tokenizing the Sentences
    - Find Weighted Frequency of Occurrence
    - Replace Words by Weighted Frequency in Original Sentences
    - Sort Sentences in Descending Order of Sum

### Steps:
1. Install the necessary packages
2. Run preprocess.py in the `Gradesum/model/`
3. Run train.py in the `Gradesum/model/`
4. Run app.py
