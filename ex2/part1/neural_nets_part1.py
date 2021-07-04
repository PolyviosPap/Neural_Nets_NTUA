#!/usr/bin/env python
# coding: utf-8

# # Εργαστηριακή Άσκηση 2. Μη επιβλεπόμενη μάθηση. 
# 
# ## Σύστημα συστάσεων βασισμένο στο περιεχόμενο
# 
# ### Παπακωνσταντίνου Πολύβιος 03114892
# ### Πατρής Νικόλαος 03114861

# ## Εισαγωγή του Dataset

# Το σύνολο δεδομένων με το οποίο θα δουλέψουμε είναι βασισμένο στο [Carnegie Mellon Movie Summary Corpus](http://www.cs.cmu.edu/~ark/personas/). Πρόκειται για ένα dataset με περίπου 40.000 περιγραφές ταινιών. Η περιγραφή κάθε ταινίας αποτελείται από τον τίτλο της, μια ή περισσότερες ετικέτες που χαρακτηρίζουν το είδος της ταινίας και τέλος τη σύνοψη της υπόθεσής της. Αρχικά εισάγουμε το dataset στο dataframe `df_data_1`: 

# In[1]:


import pandas as pd

dataset_url = "https://drive.google.com/uc?export=download&id=1PdkVDENX12tQliCk_HtUnAUbfxXvnWuG"
# make direct link for drive docs this way https://www.labnol.org/internet/direct-links-for-google-drive/28356/
df_data_1 = pd.read_csv(dataset_url, sep='\t',  header=None, quoting=3, error_bad_lines=False)


# Κάθε ομάδα θα δουλέψει σε ένα μοναδικό υποσύνολο 5.000 ταινιών (διαφορετικό dataset για κάθε ομάδα) ανάλογα με τον αριθμό (seed) που το έχει ανατεθεί στο `spreadsheets`. Στην περίπτωση μας ο αριθμός είναι 30.

# In[2]:


import numpy as np

# βάλτε το seed που αντιστοιχεί στην ομάδα σας
team_seed_number = 30

movie_seeds_url = "https://drive.google.com/uc?export=download&id=1NkzL6rqv4DYxGY-XTKkmPqEoJ8fNbMk_"
df_data_2 = pd.read_csv(movie_seeds_url, header=None, error_bad_lines=False)

# επιλέγεται 
my_index = df_data_2.iloc[team_seed_number,:].values

titles = df_data_1.iloc[:, [2]].values[my_index] # movie titles (string)
categories = df_data_1.iloc[:, [3]].values[my_index] # movie categories (string)
bins = df_data_1.iloc[:, [4]]
catbins = bins[4].str.split(',', expand=True).values.astype(np.float)[my_index] # movie categories in binary form (1 feature per category)
summaries =  df_data_1.iloc[:, [5]].values[my_index] # movie summaries (string)
corpus = summaries[:,0].tolist() # list form of summaries


# - Ο πίνακας **titles** περιέχει τους τίτλους των ταινιών. Παράδειγμα: 'Sid and Nancy'.
# - O πίνακας **categories** περιέχει τις κατηγορίες (είδη) της ταινίας υπό τη μορφή string. Παράδειγμα: '"Tragedy",  "Indie",  "Punk rock",  "Addiction Drama",  "Cult",  "Musical",  "Drama",  "Biopic \[feature\]",  "Romantic drama",  "Romance Film",  "Biographical film"'. Παρατηρούμε ότι είναι μια comma separated λίστα strings, με κάθε string να είναι μια κατηγορία.
# - Ο πίνακας **catbins** περιλαμβάνει πάλι τις κατηγορίες των ταινιών αλλά σε δυαδική μορφή ([one hot encoding](https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f)). Έχει διαστάσεις 5.000 x 322 (όσες οι διαφορετικές κατηγορίες). Αν η ταινία ανήκει στο συγκεκριμένο είδος η αντίστοιχη στήλη παίρνει την τιμή 1, αλλιώς παίρνει την τιμή 0.
# - Ο πίνακας **summaries** και η λίστα **corpus** περιλαμβάνουν τις συνόψεις των ταινιών (η corpus είναι απλά ο summaries σε μορφή λίστας). Κάθε σύνοψη είναι ένα (συνήθως μεγάλο) string. Παράδειγμα: *'The film is based on the real story of a Soviet Internal Troops soldier who killed his entire unit  as a result of Dedovschina. The plot unfolds mostly on board of the prisoner transport rail car guarded by a unit of paramilitary conscripts.'*
# - Θεωρούμε ως **ID** της κάθε ταινίας τον αριθμό γραμμής της ή το αντίστοιχο στοιχείο της λίστας. Παράδειγμα: για να τυπώσουμε τη σύνοψη της ταινίας με `ID=99` (την εκατοστή) θα γράψουμε `print(corpus[99])`.

# In[3]:


ID = 99
print(titles[ID])
print(categories[ID])
print(catbins[ID])
print(corpus[ID])


# # Εφαρμογή 1. Υλοποίηση συστήματος συστάσεων ταινιών βασισμένο στο περιεχόμενο
# <img src="http://clture.org/wp-content/uploads/2015/12/Netflix-Streaming-End-of-Year-Posts.jpg" width="50%">

# Η πρώτη εφαρμογή που θα αναπτύξετε θα είναι ένα [σύστημα συστάσεων](https://en.wikipedia.org/wiki/Recommender_system) ταινιών βασισμένο στο περιεχόμενο (content based recommender system). Τα συστήματα συστάσεων στοχεύουν στο να προτείνουν αυτόματα στο χρήστη αντικείμενα από μια συλλογή τα οποία ιδανικά θέλουμε να βρει ενδιαφέροντα ο χρήστης. Η κατηγοριοποίηση των συστημάτων συστάσεων βασίζεται στο πώς γίνεται η επιλογή (filtering) των συστηνόμενων αντικειμένων. Οι δύο κύριες κατηγορίες είναι η συνεργατική διήθηση (collaborative filtering) όπου το σύστημα προτείνει στο χρήστη αντικείμενα που έχουν αξιολογηθεί θετικά από χρήστες που έχουν παρόμοιο με αυτόν ιστορικό αξιολογήσεων και η διήθηση με βάση το περιεχόμενο (content based filtering), όπου προτείνονται στο χρήστη αντικείμενα με παρόμοιο περιεχόμενο (με βάση κάποια χαρακτηριστικά) με αυτά που έχει προηγουμένως αξιολογήσει θετικά.
# 
# Το σύστημα συστάσεων που θα αναπτύξετε θα βασίζεται στο **περιεχόμενο** και συγκεκριμένα στις συνόψεις των ταινιών (corpus). 
# 

# ## Μετατροπή σε TFIDF
# 
# Το πρώτο βήμα θα είναι λοιπόν να μετατρέψετε το corpus σε αναπαράσταση tf-idf:

# In[4]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(corpus)
corpus_tf_idf = vectorizer.transform(corpus)


# Η συνάρτηση [TfidfVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) όπως καλείται εδώ **δεν είναι βελτιστοποιημένη**. Οι επιλογές των μεθόδων και παραμέτρων της μπορεί να έχουν **δραματική επίδραση στην ποιότητα των συστάσεων** και είναι διαφορετικές για κάθε dataset. Επίσης, οι επιλογές αυτές έχουν πολύ μεγάλη επίδραση και στη **διαστατικότητα και όγκο των δεδομένων**. Η διαστατικότητα των δεδομένων με τη σειρά της θα έχει πολύ μεγάλη επίδραση στους **χρόνους εκπαίδευσης**, ιδιαίτερα στη δεύτερη εφαρμογή της άσκησης. 

# In[5]:


print(corpus_tf_idf.shape)


# Παρατηρούμε ότι το μέγεθος του **tf-idf** πίνακα στην περίπτωση του default vectorizer είναι δραματικά μεγάλο. Με μια γρήγορη ματιά φαίνεται ότι λέξεις χωρίς κανένα ουσιαστικό νοηματικό περιεχόμενο επίλεγονται ως features του vector. 

# In[6]:


print(vectorizer.get_feature_names()[:10], vectorizer.get_feature_names()[48954:])


# *Unicodes, αριθμοί, σημειά στήξης* και πολλά αλλά αντιστοιχιζόνται σε features και επιβαρύνουν ιδιαίτερα την ποιότητα των συστάσεων που θα προκύψουν από το dataset μας. Οι λύσεις που έχουμε στην διάθεση μας είναι οι εξής: 
# 
# - δημιουργία ένος δικού μας tokenizer που θα ανταποκρύνεται καλύτερα στο dataset μας
# - απόρριψη λέξεων που δεν προσφέρουν σημασιολογική αξία στο κείμενο, όπως κύρια ονόματα ή λέξεις όπως "the", "a", "to", "and", "he", "she" κοκ. (stopwords)
# - ενδεχομένως Stemming & Lemmatization 
# - και τέλος κατάλληλο tuning τως παραμέτρων του TfidfVectorizer().

# In[7]:


import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

porter_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()

stop_words = list(stopwords.words('english'))
punctuation = list(string.punctuation)
numbers = list("0123456789")

invalid = set(numbers + punctuation)

def thorough_filter(words):
    filtered_words = []
    for word in words:
        pun = []
        valid_flag = True
        
        for letter in word:
            if letter in invalid:
                valid_flag = False
                break
                
        if valid_flag:
            filtered_words.append(word)
    return filtered_words

def tokenize(corpus):
    words = nltk.word_tokenize(corpus.lower())
    filtered_words = [word for word in words if word not in stop_words]
    filtered_words = thorough_filter(filtered_words)
    porter_stemmer = PorterStemmer()
    stem_words = [porter_stemmer.stem(word) for word in filtered_words]
    return stem_words


# In[8]:


vectorizer = TfidfVectorizer(max_df=0.4, min_df=5, tokenizer=tokenize, strip_accents=ascii)
corpus_tf_idf = vectorizer.fit_transform(corpus)


# In[9]:


print(corpus_tf_idf.shape)


# In[10]:


print(vectorizer.get_feature_names()[:10], vectorizer.get_feature_names()[8856:])


# Ως τελευταία προσπάθεια μπορούμε να "πειράξουμε" τα όρια των επιτρεπτών συχνοτήτων των λέξεων `max_df, min_df`. Θα ασχοληθούμε πρώτα με το άνω όριο **(max_df)** και έπειτα με το **(min_df)**.

# In[11]:


vectorizer = TfidfVectorizer(max_df=0.2, min_df=5, tokenizer=tokenize, strip_accents=ascii)
corpus_tf_idf = vectorizer.fit_transform(corpus)
print(corpus_tf_idf.shape)

vectorizer = TfidfVectorizer(max_df=0.4, min_df=5, tokenizer=tokenize, strip_accents=ascii)
corpus_tf_idf = vectorizer.fit_transform(corpus)
print(corpus_tf_idf.shape)

vectorizer = TfidfVectorizer(max_df=0.6, min_df=5, tokenizer=tokenize, strip_accents=ascii)
corpus_tf_idf = vectorizer.fit_transform(corpus)
print(corpus_tf_idf.shape)


# Παρατηρούμε ότι η μέγιστη τιμή επηρεάζει ελάχιστα το πλήθος των features του vectorizer, καθώς οι περισσότεροι όροι δεν φτάνουν σε τόσα υψηλά (~0.2) ποσοστά συγκέντρωσης στο dataset. Το μόνο που μένει να παραμετροποιηθεί είναι το κάτω όριο της μετρικής document freq.

# In[12]:


vectorizer = TfidfVectorizer(max_df=0.4, min_df=5, tokenizer=tokenize, strip_accents=ascii)
corpus_tf_idf = vectorizer.fit_transform(corpus)
print(corpus_tf_idf.shape)

vectorizer = TfidfVectorizer(max_df=0.4, min_df=10, tokenizer=tokenize, strip_accents=ascii)
corpus_tf_idf = vectorizer.fit_transform(corpus)
print(corpus_tf_idf.shape)

vectorizer = TfidfVectorizer(max_df=0.4, min_df=15, tokenizer=tokenize, strip_accents=ascii)
corpus_tf_idf = vectorizer.fit_transform(corpus)
print(corpus_tf_idf.shape)

vectorizer = TfidfVectorizer(max_df=0.4, min_df=20, tokenizer=tokenize, strip_accents=ascii)
corpus_tf_idf = vectorizer.fit_transform(corpus)
print(corpus_tf_idf.shape)


# Το μόνο που καταφέραμε σε αυτό το στάδιο ήταν να ελέγχουμε ορισμένες **καλές** τιμές για τις παραμέτρους μας. Θα πρέπει να επαλανάβουμε ορισμένα από τα βήματα αυτά όταν θα θέσουμε τον Vectorizer στο σύστημα συστάσεων.

# ## Υλοποίηση του συστήματος συστάσεων
# 
# Το σύστημα συστάσεων που θα φτιάξουμε θα είναι μια συνάρτηση `content_recommender` με δύο ορίσματα `target_movie` και `max_recommendations`. Στην `target_movie` περνάμε το ID μιας ταινίας-στόχου για την οποία μας ενδιαφέρει να βρούμε παρόμοιες ως προς το περιεχόμενο (τη σύνοψη) ταινίες, `max_recommendations` στο πλήθος.
# 
# - Για τις `max_recommendations` ταινίες (πλην της ίδιας της ταινίας-στόχου που έχει cosine similarity 1 με τον εαυτό της) με τη μεγαλύτερη ομοιότητα συνημιτόνου (σε φθίνουσα σειρά), τυπώστε σειρά σύστασης (1 πιο κοντινή, 2 η δεύτερη πιο κοντινή κλπ), id, τίτλο, σύνοψη, κατηγορίες (categories)

# In[13]:


corpus_tf_idf.shape[0]


# In[14]:


import scipy as sp

def content_recommender(target_movie, max_recommendations):
    similarity = []
    movies_tfidf = corpus_tf_idf.toarray()
    
    for val in movies_tfidf:
        similarity.append(1.0-sp.spatial.distance.cosine(movies_tfidf[target_movie], val))
    
    similarity = np.argsort(similarity, kind='quicksort')[::-1]
    
    return similarity[1:max_recommendations+1]

target_movie = 749
max_recommendations = 2

def report_movie(title, movie_id):
    print("""
    -- {0}[{1}] {2} --
    Categories: {3}
    Summary: {4:<10}...
    {5}
    """.format(title, movie_id, titles[movie_id], categories[movie_id], corpus[movie_id][:400], "-" * 100))
    
def report_recommendations(target_movie, max_recommendations):
    report_movie("Target Movie", target_movie)
    
    i = 1
    for rec in content_recommender(target_movie, max_recommendations):
        report_movie("Recommended #{0}".format(i) , rec)
        i += 1
    


# In[15]:


report_recommendations(742, 2)


# ## Βελτιστοποίηση
# 
#    Αφού υλοποιήσαμε τη συνάρτηση `content_recommender` θα την χρησιμοποιήσουμε για να βελτιστοποιήσουμε περαιτέρω τον `TfidfVectorizer`. Επειδή ήδη στο προηγούμενο παράδειγμα οι ταινίες που μας ___πρότεινε___ είχαν κοινό νοηματικό περιεχόμενο (Zombies, Thriller) θα προσπαθήσουμε να αυξήσουμε την τιμή του `max_recommendations` για να δούμε πότε αρχίζουμε να έχουμε απόκληση. Ως πρώτο βήμα θα αποδείξουμε ότι η χρήση του tokenizer που γράψαμε παρέχει καλύτερα αποτελέσματα απο εκείνα του default που κάνει χρήση ο `TfidfVectorizer`.

# In[16]:


vectorizer = TfidfVectorizer(max_df=0.2, min_df=20, stop_words='english')
corpus_tf_idf = vectorizer.fit_transform(corpus)
print('Default Vectorizer Shape: ', corpus_tf_idf.shape)

report_recommendations(1753, 15)


# In[17]:


vectorizer = TfidfVectorizer(max_df=0.4, min_df=10, tokenizer=tokenize, strip_accents=ascii)
corpus_tf_idf = vectorizer.fit_transform(corpus)
print('Better Vectorizer Shape: ', corpus_tf_idf.shape)

report_recommendations(1753, 15)


#     Ήδη από την 9η ταινία η default υλοποίηση του Vectorizer αρχίζει να ξεφεύγει απο το context της target_movie.

# Ως πρώτη απόπειρα θα προσπαθήσουμε να εισάγωγουμε μαζί με μονό **tokens**, και ορισμένα _n-grams_, ώστε λέξεις που βρίσκονται κοντά στο κείμενο και χαρακτηρίζονται από κοινό σημασιολογικό περιέχομενο να αποτελούν και αυτές πλέον features. Το πρόβλημα με της χρήση των _ngrams_ είναι ότι μεγαλώνουν αρκέτα το `corpus_tf_idf`, χωρίς πάντα να έχουν δραματικά αποτελέσματα.

# In[18]:


vectorizer = TfidfVectorizer(max_df=0.4, min_df=10, ngram_range=(1,3), tokenizer=tokenize, strip_accents=ascii)
corpus_tf_idf = vectorizer.fit_transform(corpus)


# In[19]:


report_recommendations(1753, 15)


#       Τα αποτελέσματα δεν είναι ιδιαίτερα ικανοποιητικά καθώς φαίνεται οι προτάσεις που μας επιστρέφει να μην διαφέρουν κατά πολύ με αυτές του αρχικού παρόλο που καταλήξαμε με σχεδόν 25% μεγαλύτερο μέγεθος πίνακα. 

# Στα παρακάτω κελιά θα προσπαθήσομυε να τροποποιήσουμε τις παραμέτρους του `TfidfVectorizer`, ώστε να καταφέρουμε να πετύχουμε ακριβέστερα αποτελέσματα και ενδεχομένως μικρότερο μέγεθος στον τελικό πίνακα. Θα ξεκινήσουμε κανονικοποιώντας τα διανύσματα και θα προσθέσουμε έπειτα θα εφαρμόσουμε _sublinear tf scaling_.

# In[20]:


vectorizer = TfidfVectorizer(max_df=0.4, min_df=20, tokenizer=tokenize, strip_accents=ascii, norm='l1')
corpus_tf_idf = vectorizer.fit_transform(corpus)
print(corpus_tf_idf.shape)


# In[21]:


report_recommendations(1753, 10)


# In[22]:


vectorizer = TfidfVectorizer(max_df=0.4, min_df=20, tokenizer=tokenize, strip_accents=ascii, norm='l2')
corpus_tf_idf = vectorizer.fit_transform(corpus)
print(corpus_tf_idf.shape)


# In[23]:


report_recommendations(1753, 10)


#     Ούτε η κανονικοποίηση των διανυσμάτων φάνηκε να έχει κάποια ιδιαίτερη επίδραση στα αποτελέσματα. Θα προχωρήσουμε με την εφα-ρμογή του sublinear scaling.

# In[24]:


vectorizer = TfidfVectorizer(max_df=0.4, min_df=20, tokenizer=tokenize, strip_accents=ascii, norm='l2', sublinear_tf=False)
corpus_tf_idf = vectorizer.fit_transform(corpus)
print(corpus_tf_idf.shape)


# In[25]:


report_recommendations(1753, 10)


# Αφού πλέον καμία τεχνική δεν βοήθησε στα αποτελέσματα θα προσπαθήσουμε με μια γρήγορη επισκόπηση των features να κρίνουμε αν υπάρχουν "λέξεις" που δεν προσθέρουν ιδιαίτερα στο επιτυχημένο αποτέλεσμα παρά μόνο να επιβαρύνουν τον τελικό μας πίνακα με επιπλέον μέγεθος.

# In[26]:


print(vectorizer.get_feature_names())


# Το μόνο που παρατηρούμε είναι ορισμένα ονόματα όπως Dan, Dave, David, Christopher, Edward. Δυστυχώς τα όνομα δεν διαθέτουν κάποιο νοηματικό βάρος στις περιλήψεις των κειμένων για αυτό είναι καλό να μην υπάρχουν ως features. Για να το πετύχουμε αυτό θα χρειαστεί να εισάγουμε ορισμένες επιπλέον *stop_words* στον tokenizer μας.

# In[27]:


from nltk.corpus import names

invalid_names = []
for name in names.words():
    invalid_names.append(name.lower())

stop_words += invalid_names


# In[28]:


vectorizer = TfidfVectorizer(max_df=0.4, min_df=25, tokenizer=tokenize, strip_accents=ascii)
corpus_tf_idf = vectorizer.fit_transform(corpus)
print(corpus_tf_idf.shape)


# ## Ποιοτική ερμηνεία
# 
# Θα δώσουμε 10 παραδείγματα (IDs) από τη συλλογή σας που επιστρέφουν καλά αποτελέσματα μέχρι `max_recommendations` (5 και παραπάνω) και σημειώστε συνοπτικά ποια είναι η θεματική που ενώνει τις ταινίες.

# In[30]:


# Χωρός και εφηβικές σχέσεις
report_recommendations(124, 7)


# In[32]:


# Πόλεμος και εξέλιξη ιστορίας, Χώρες της Άπω Ανατολής
report_recommendations(64, 10)


# In[33]:


# Zombies, Supernatural και Horror στοιχεία
report_recommendations(1753, 10)


# In[34]:


# Αεροπλάνο, πιλότοι, αερομαχίες
report_recommendations(570, 10)


# In[35]:


# Ληστείες λαθρεμπόρειο διαμαντιών
report_recommendations(4243, 10)


# In[36]:


# Cowboys, άγρια Δύση, άλογα
report_recommendations(2032, 10)


# In[37]:


# Περιπέτεια και (πρωτόγενες) φυλές
report_recommendations(3065, 10)


# In[38]:


# Κολλέγιο, ποδοσφαιρικές ομάδες κολλεγίου
report_recommendations(1312, 10)


# In[39]:


# Frankenstein, υπερφυσικά τέρατα, μυστήριο
report_recommendations(4965, 10)


# Η απόδοση μιας συνάρτησης σαν τη δική μας, φαίνεται να επηρεάζεται πρώτα απ' όλα απ' την ποιότητα των κειμένων στα οποία δουλεύει.
# 
# Όπως είχαμε προβλέψει απ' την αρχή, ταινίες με πολύ μικρή υπόθεση (1-2 γραμμές) δεν έδιναν αρκετές και σημαντικές πληροφορίες για να βρούμε προτεινόμενες.
# 
# Στο άλλο άκρο, εκτενείς υποθέσεις πλάτειαζαν και "μπέρδευαν" τη συνάρτηση στο να ξεχωρίσει την ουσία της υπόθεσης.
# 
# Τέλος, σημειώνεται το προφανές: δεν μπορούσαμε να "αντιμετωπίσουμε" λέξεις με διπλό νόημα. Για παράδειγμα δοκιμάζοντας ταινίες Χριστουγέννων, υπήρχε σύγχηση μεταξύ του "snow" (χιόνι) και "snow white" (χιονάτη).

# ## Persistence αντικειμένων με joblib.dump
# 
# Ας αποθηκεύσουμε το `corpus_tf_idf` και στη συνέχεια ας το ανακαλέσουμε.

# In[ ]:


from sklearn.externals import joblib
joblib.dump(corpus_tf_idf, 'corpus_tf_idf.pkl') 


# 
# 
# Μπορείτε με ένα απλό `!ls` να δείτε ότι το αρχείο `corpus_tf_idf.pkl` υπάρχει στο filesystem σας (== persistence):

# In[ ]:


get_ipython().system('ls -lh | grep corpus*')

