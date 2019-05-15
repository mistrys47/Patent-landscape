import os
from gensim.summarization import summarize
from builtins import str

def summ_it():
    total = 0
    for file in os.listdir('topics'):
        with open('topics/' + file, 'r') as f:
            summ = summarize(str(f.read().replace('\n',' ')), word_count=75)
            #with open('summ_gensim/' + file.split('.')[0] + '.txt', 'w') as fw:
            print(file)
            print(summ)
            #total += len(summ)

summ_it()


#code 2
def summ_it():
    total = 0
    for i in range(20):
        try:
            summ = summarize(str(training_df['abstract_text'][i]), word_count=25)
        except:
            summ = "error"
        #with open('summ_gensim/' + file.split('.')[0] + '.txt', 'w') as fw:
        print(i+1)
        print('original :')
        print(len(training_df['abstract_text'][i]))
        print('summarized :')
        print(len(summ))
        total += len(summ)
summ_it()