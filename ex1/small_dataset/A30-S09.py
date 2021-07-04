#!/usr/bin/env python
# coding: utf-8

# ## Μέρος Α 
# #### Στοιχεία Ομάδας
# 
# Παπακωνσταντίνου Πολύβιος 03114892
# 
# Πατρής Νικόλαος 03114861

# In[1]:


import warnings 
warnings.filterwarnings('ignore')


# ## Μέρος Β

# #### Εισαγωγή του dataset  
#   
# Θα εισάγουμε και θα διαβάζουμε το csv αρχείο `sonar.all-data` με την χρήση της read_csv και option "header=None" γιατί η πρώτη γραμμή περιέχει δεδομένα.

# In[2]:


import pandas as pd
df = pd.read_csv("sonar.all-data", header=None)


# #### Παρουσίαση του dataset.
# Στόχος του dataset είναι να εκπαιδεύσει ένα νευρωνικό δίκτυο να διακρίνει ηχοεντοπιστικά σήματα (sonars) τα οποία αντανακλώνται είτε απο μεταλικούς κυλίνδρους είτε από κυλινδρικούς βράχους/πέτρες.
# 
# Έχουμε στην διάθεση μας **208** δείγματα, τα πρώτα 111 αφορούν ανακλάσεις σε μεταλλικά αντικείμενα (κυλίνδρρους) υπό διαφορετικές γωνίες και συνθήκες. Τα υπολοίπα (97) αφορούν τον δεύτερο τύπο ανακλάσης, σε κυλινδρικούς βράχους, με παρόμοιες παραλλαγές μέτρησης όπως τα πρώτα.
# 
# Κάθε δείγμα αποτελείται απο 60 αριθμούς (δείγματα) στο διάστημα $[0.0,1.0]$, ένω κάθε αριθμός αναπαριστά την ενέργεια που εμφανίζεται σε διαφορετικά φάσματα συχνοτήτων μέσα σε ένα χρονικό διάστημα.

# ####  Αριθμός δειγμάτων και χαρακτηριστικών, είδος χαρακτηριστικών. Υπάρχουν μη διατεταγμένα χαρακτηριστικά και ποια είναι αυτά;

# In[3]:


print("# Samples: {}, # Features: {}".format(df.shape[0], df.shape[1]-1))
print(df.head())


# Όπως προείπαμε διαθέτουμε 208 δείγματα (111 μέταλο, 97 πέτρα) και 60 χαρακτηριστικά.
# 
# **να γράψουμε ότι δεν υπάρχουν μη διατεταγμένα χαρακτηριστικά αφού το μόνο μη διατεταγμένο είναι το output(;).

# #### Υπάρχουν επικεφαλίδες;  Αρίθμηση γραμμών;
# Τα δεδομένα είναι σε `row` μορφή μέσα στο αρχείο, χωρίς επικεφαλίδες και αριθμήσεις γραμμών/δειγμάτων.

# #### Ποιες είναι οι ετικέτες των κλάσεων και σε ποια κολόνα βρίσκονται; 
# H 61η στήλη είναι εκείνη που περιγράφει την κλάση κάθε δείγματος: **M** (metal) ή __R__ (rock).

# In[4]:


import numpy as np

labels = df.iloc[:,-1].unique()
print("Unique Labels of samples: {0}".format(labels))


# #### Χρειάστηκε να κάνετε μετατροπές στα αρχεία text και ποιες?
# 
# Έχουμε ήδη αναφερθεί στην μορφή του αρχείου δεδομένων, και στον τρόπο με τον οποία αναγράφονται τα δείγματα. Ωστόσο το μόνο που μένει να τροποποιηθεί είναι η μορφή των labels. Όπως γνωρίζουμε, ένα νευρωνικό δίκτυο δεν ειναι σε θέση να παράγει στην έξοδο του κάποια συμβολοσειρά, αλλά αριθμούς. Για να καταφέρουμε να εκπαιδεύσουμε τον νευρωνικό θα χρειαστεί αρχικά να μετατρέψουμε τα string 'R' και 'M' σε 0 και 1 αντίστοιχα.  
# 
# Ορίζοντας ένα mapping και εφαρμόζοντας τη μέθοδο replace με το συγκεκριμένο mapping, η μετατροπή γίνεται αυτόματα σε όλα τα δείγματα.  

# In[5]:


features_df = df.iloc[:,:-1]
labels_df = df.iloc[:,-1]

mapping = {'R': 0, 'M': 1}
labels_df = labels_df.replace(mapping)


# Μπορούμε τώρα να μετατρέψουμε τα dataframes σε numpy, ώστε να είναι συμβατά με το `scikit-learn`.

# In[6]:


features = features_df.values
labels = labels_df.values


# #### Υπάρχουν απουσιάζουσες τιμές; 
# 
# Ένας απλός τρόπος να ελέγξουμε αν υπάρχουν απουσιάζουσες τιμές είναι ο παρακάτω:

# In[7]:


missing = df.isnull().values.any()
print(missing)


# Εφόσον μας επιστρέφει false, δεν υπάρχουν απουσιάζουσες τιμές.

# #### Ποιος είναι ο αριθμός των κλάσεων και τα ποσοστά δειγμάτων τους επί του συνόλου; 

# In[8]:


frequencies = np.bincount(labels)
total_samples = frequencies.sum()
percentage = (frequencies / total_samples) * 100

print("""Class frequencies: {0}
Total samples: {1}
Class percentage: {2}""".format(frequencies, total_samples, percentage))


# Παρατηρούμε πως και οι 2 κλάσεις είναι αρκετά κοντά στο 50%, συνεπώς το dataset μας είναι ισορροπημένο.

# #### Διαχωρισμός σε train και test sets.
# 
# Ο διαχωρισμός του dataset σε train, test θα γίνει με την βοήθεια της έτοιμης ρουτίνας του sklearn. Όσο για την παράμετρο `random_state` μιας και η συμβολή της στον διαχωρισμό είναι ελάχιστή, την αρχικοποιήσαμε στην τιμή 23.

# In[9]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, random_state=23)

print("Split: Train {0}, Test {1}".format(np.bincount(y_train), np.bincount(y_test)))


# Παρατηρούμε ωστόσο ότι η συμβολή της `random_state` στον διαχωρισμό αν και μικρή μπορεί να δημιουργήσει *ανισορροπίες* στα δεδομένα που θα παρέχουμε για εκπαίδευση. Ιδιώς στην περίπτωση μας εκπαιδεύουμε την ταξινομητή μας σχεδόν $30\%$ περισσότερο με `M samples`.  
# 
# Επομένως, επιλέξαμε να ενεργοποιήσουμε και την ικάνοτητα __*statify*__ ώστε να μας παράξει ένα πιο ισορροπημένο διαχωρισμό μεταξύ των δύο κλάσεων. 

# In[10]:


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels)

print("Split (Stratified): Train {0}, Test {1}".format(np.bincount(y_train), np.bincount(y_test)))


# ## Μέρος Γ
# 
# Στο τρίτο μέρος καλούμαστε να εκπαιδεύσουμε τους classifiers με default τιμές που επιλέγουμε εμείς. Ο μόνος παραμετρικός ταξινομητής που έχουμε να εξετάσουμε είναι ο `KNeighborsClassifier` οπότε η τιμή που θα επιλέξουμε για τον αριθμό των γειτόνων, μπορεί να μην οδηγήσει στα βέλτιστα αποτελέσματα.
# 
# Αρχικά ορίζουμε όλους του $Dummy\ Classifiers$ και τους εκπαιδεύουμε  με το $train\_set$.

# In[118]:


from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Initialize some dummies classifiers

dc_uniform = DummyClassifier(strategy="uniform")
dc_constant_0 = DummyClassifier(strategy="constant", constant=0)
dc_constant_1 = DummyClassifier(strategy="constant", constant=1)
dc_most_frequent = DummyClassifier(strategy="most_frequent")
dc_stratified = DummyClassifier(strategy="stratified")


# In[325]:


# Train each dummy classifier.

dc_uniform.fit(X_train, y_train)
dc_constant_0.fit(X_train, y_train)
dc_constant_1.fit(X_train, y_train)
dc_most_frequent.fit(X_train, y_train)
dc_stratified.fit(X_train, y_train)


# In[326]:


# Predict values based on test set

pred_uni = dc_uniform.predict(X_test)
pred_const_0 = dc_constant_0.predict(X_test)
pred_const_1 = dc_constant_1.predict(X_test)
pred_freq = dc_most_frequent.predict(X_test)
pred_strat = dc_stratified.predict(X_test)


# In[327]:


from sklearn.neighbors import KNeighborsClassifier
# kNN Classifier, default k=5

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
pred_knn = knn.predict(X_test)


# #### Confusion matrix - Bar Plots F1 (micro/macro) Score
# 
# Όλες οι προβλέψεις των ταξινομονητών υπολογίστικαν και αποθηκεύτηκαν σε ξεχωριστές μεταβλητές ώστε να αποφύγουμε τυχόν overwrites. Το επόμενο βήμα είναι να υπολογίσουμε τις κατάλληλες μετρικές και τους πίνακες σύγχυσης που θα μας βοηθήσουν στην ανάλυση των μοντέλων μας. Αποφασίσαμε να γράψουμε ορισμένες ρουτίνες για να αυτοματοποιήσουμε την διαδικασία, και να βοήθησουν στην καλύτερη αναπαράσταση των δεδομένων.
# 
# `cnf_score_report`: Μια απλή συνάρτηση που την καλούμε με $y\_test,\ y\_pred$, και επιστρέφει $recall,\ precision, f1\ scores,\ confusion\ matrices$.

# In[328]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

def cnf_score_report(y_test, y_pred):
    cnf_matrix = confusion_matrix(y_test, y_pred)
    
    f1_micro = precision_recall_fscore_support(y_test, y_pred, average='micro')[2]
    recall_micro = precision_recall_fscore_support(y_test, y_pred, average='micro')[1]
    precision_micro = precision_recall_fscore_support(y_test, y_pred, average='micro')[0]

    f1_macro = precision_recall_fscore_support(y_test, y_pred, average='macro')[2]
    recall_macro = precision_recall_fscore_support(y_test, y_pred, average='macro')[1]
    precision_macro = precision_recall_fscore_support(y_test, y_pred, average='macro')[0]
    
    return (cnf_matrix, round(precision_micro,4), round(recall_micro,4), round(f1_micro,4), round(precision_macro,4), round(recall_macro,4), round(f1_macro,4))


# Η επόμενη συνάρτηση αφορά την παρουσίαση των πινάκων σύγχυσης. Θεωρήσαμε ότι η εμφάνιση τους σαν απλούς `numpy arrays` δεν θα βοηθούσε ιδιαίτερα στην μελέτη, οπότε προχωρήσαμε στην αναζήτηση μιας πιο εξειδικευμένης συνάρτησης, που με συνδυασμό με την βιβλιοθήκη `matplotlib`, θα τους εμφάνιζε με τρόπο παρόμοιο με αυτόν που είδαμε στο εργαστήριο. Ο παρακάτω κώδικας, είναι μία __ελαφρώς__ τροποποιημένη μορφή του αρχικού που βρήκαμε στην σελίδα του sklearn.

# In[329]:


"""
Confusion Matrix Plotter from sklearn
https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
"""

import itertools
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
    else:
        pass
#         print('Confusion matrix, without normalization')

#     print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="black" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    


# Η επόμενη συνάρτηση κάνει χρήση της κλάσης που ανήκει κάθε ταξινομητής, και διαμορφώνει το κατάλληλο string, που θα χρησιμοποιηθεί ως όνομα του `x-axis` για κάθε ένα ταξινομητή τόσο στους πίνακες σύγχυσης όσο και στο barplot.

# In[330]:


def transfrom_title(class_type):
    split_title = class_type.replace(")", "").split("(")
    clf_name = split_title[0]
    
    if (clf_name == "DummyClassifier"):
        return clf_name + split_title[1].split(",")[2]
    elif (clf_name == "KNeighborsClassifier"):
        return clf_name + split_title[1].split(",")[5]


# Η `plot-report` χρησιμοποιεί τις παραπάνω τρεις συναρτήσεις που σχολιάσαμε και εμφανίζει σε διαφορετικό figure τους confusion matrices, για κάθε ένα ταξινομητή. Ως όρισμα μπορεί να δεχτεί και έναν μεμονωμένο ταξινομητή αλλά και μια λίστα εξ αυτών διαμορφώνοντας κατάλληλα το input για να δουλέψει η for loop. Επίσης δέχεται και μια λίστα ονομάτων, ώστε να αντιστοιχίσει κάθε binary κλάση, με το ονόμα που την περιγράφει.

# In[514]:


def plot_report(clfs, X_test, y_test, classes):
    if not isinstance(clfs, list):
        clfs = [clfs]
    
    for i, clf in enumerate(clfs):
        plt.figure(i)
        y_pred = clf.predict(X_test)
        cnf_matrix, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro = cnf_score_report(y_test, y_pred)
        
        title = transfrom_title(str(clf))
        title = "{0} \n [Micro] Precision: {1} Recall: {2}, F1: {3} \n [Macro] Precision: {4} Recall: {5}, F1: {6}".format(title, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro)
        plot_confusion_matrix(cnf_matrix, title=title, classes=classes)
        
    plt.show()
        


# In[332]:


clfs = [dc_uniform, dc_constant_0, dc_constant_1, dc_most_frequent, dc_stratified, knn]
plot_report(clfs, X_test, y_test, ["Rock", "Metal"])


# Η `barplot_score` εμφανίζει τα δύο ζητούμενα bar plots για τις δύο μετρικές $f1\_micro,\ f1\_macro$. Επειδή τα ονόματα των ταξινομητών είναι αρκετά μεγάλα για τον x-άξονα, στην περίπτωση των __Dummies__ εμφανίζουμε μόνο την στρατηγική που ακολουθεί, ενώ στον __KNeighbors__ των αριθμό των γειτόνων.

# In[515]:


def barplot_score(clfs, X_test, y_test):
    if not isinstance(clfs, list):
        clfs = [clfs]
    
    micro_l = []
    macro_l = []
    clf_name = []
    
    for clf in clfs:
        _, _, _, f1_micro, _, _, f1_macro = cnf_score_report(y_test, clf.predict(X_test))
        
        micro_l.append(f1_micro)
        macro_l.append(f1_macro)
        clf_name.append(transfrom_title(str(clf)).split()[1])
        
    y_pos = np.arange(len(clf_name))
    
    plt.figure(1)
    plt.bar(y_pos, micro_l, align='center', alpha=0.9)
    plt.xticks(y_pos, clf_name, rotation=45)
    plt.ylabel('Score')
    plt.title('f1 micro')
    plt.tight_layout()
    
    plt.figure(2)
    plt.bar(y_pos, macro_l, align='center', alpha=0.9)
    plt.xticks(y_pos, clf_name, rotation=45)
    plt.ylabel('Score')
    plt.title('f1 macro')
    plt.tight_layout()


# In[334]:


barplot_score(clfs, X_test, y_test)


# Καταρχάς παρατηρούμε πως για average = 'micro', οι τιμές των `precision` και `recall` (και προφανώς του f1) είναι ίδιες για κάθε classifier ξεχωριστά.  
# 
# Αυτό συμβαίνει επειδή στο micro averaging, τα `FalsePositive` είναι ίσα με τα `FalseNegative` στον τύπο αφού υπολογίζονται __globally__.  
# 
# Για παράδειγμα αν ένα δείγμα κλάσης Α υπολογίστηκε ως Β, για την κλάση Α θα είναι `FalseNegative`, ενώ για την Β `FalsePositive`.
# Οπότε το αποτέλεσμα των recall και presicion προφανώς __θα ισούται__.
# 
# $$
# Precision_{micro} = \frac{TP_1 + TP2}{TP_1 + FP_1 + TP_2 + FP_2} \\
# Recall_{micro} = \frac{TP_1 + TP2}{TP_1 + FN_1 + TP_2 + FN_2}
# $$
# 
# Αντίθετα, στο macro averaging υπολογίζουμε τα recall και precision για κάθε κλάση ξεχωριστά και έπειτα τα συνυπολογίζουμε.
# 
# 
# $$
# Precision_{macro} = \frac{Precision_1 + Precision_2}{2} \\
# Recall_{macro} = \frac{Recall_1 + Recall_2}{2}
# $$
# 
# Επίσης με τον ίδιο τρόπο εξηγείται γιατί στους `constant 0`, `constant 1` και `most frequent` (όπου για μια κλάση έχουμε πάρα πολλά
# FalseNegative/FalsePositive, ενώ για την άλλη ελάχιστα) το macro averaging υπολογίζει σημαντικά χαμηλότερες τιμές (εκτός
# από το recall το οποίο προφανώς υπολογίζεται στο 0.5).  
# 
# $$
# Precision = \frac{TP}{TP+FP} \\
# Recall = \frac{TP}{TP+FN}
# $$
# 
# Η `stratified` από την άλλη κάνει τα predictions ανάλογα με την ισορροπία του dataset, που στην περίπτωση μας είναι σε καλό
# επίπεδο. Και πάλι όμως παρατηρούμε πως τα αποτελέσματα και για αυτή τη στρατηγική είναι αρκετά χαμηλά. 
# 
# Τέλος, οφείλουμε να αναφέρουμε πως οι `dummy classifiers` είναι αρκετά απλοί στην υλοποίηση και στη φιλοσοφία τους,
# εξού και τα χαμηλά αποτελέσματα σε σχέση με τον `knn`.
# Για παράδειγμα όταν επιλέγουμε στρατηγική `uniform`, οι προβλέψεις μας είναι τυχαίες και γι' αυτό αν ξανατρέξουμε τον
# classifier θα μας βγάλει εντελώς διαφορετικά αποτελέσματα, γεγονός που δείχνει πως είναι αρκετά αναξιόπιστη.

# ## Μέρος Δ

# Το τελευταίο και σημαντικότερο μέρος της εργασίας αφορά βεβαίως την βελτιστοποίηση των υπερπαραμέτρων. Μια διαδικασία ιδιαιτέρως απαιτητική σε χρόνο καθώς απαιτεί μεγάλο πλήθος δοκιμών για κάθε υπερπαράμετρο. Δυστυχώς ρουτίνες όπως η `GridSearchCV` δεν επιτρέπεται να χρησιμοποιηθούν για να αυτοματοποιήσουν την διαδικασία. Μετά λοιπόν απο δοκιμές αποφασίσαμε να γράψουμε την δική μας κλάση που θα βοήθησει στην εύρεση των κατάλληλων τιμών. Βεβαίως δεν μπορεί να ανταγωνιστεί συναρτήσης που παρέχει το sklearn, παρόλα αυτά οδηγεί σε σωστά αποτέλεσματα. 
# 
# Προσπαθήσαμε λοιπόν να συνχωνεύσουμε την κλάση `Pipeline` και `GridSearchCV` που παρέχει το `sklearn` σε μια ενιαία κλάση, με αντίστοιχες δυνατότητες. Πέραν λοιπόν του estimator που ορίζεται πρώτος, το δεύτερο όρισμα (grid) δέχεται ένα λεξικό (dict) στο οποίο αναφέρονται οι τιμές των παραμέτρων που θέλεις να δοκιμάσεις και συγκεκριμένα οι εξής.
# 
# - vthreshold  
# - n_components  
# - k_neighbors  
# 
# Επιτρέπεται επίσης η χρήση ενός εξισορροπιστεί δεδομένων, και συγκεκριμένα του `RandomOverSampler`, αφού μετά απο δοκιμές κρίθηκε ακατάλληλη η χρήση του `UnderSampler` καθώς αφαιρεί χρήσιμη πληροφορία, από ένα ήδη μικρό dataset. Επίσης λόγω της μορφής των features (τιμές μεταξύ [0,1]), επιλέξαμε να χρησιμοποιήσουμε μόνο των `Standard Scaler`, αφού και πάλι έπειτα απο δοκιμές οδήγησε σε καλύτερα αποτέλεσμα για τους εκτιμητές μας. Τέλος για να μπορούμε να παρέχουμε στο `Cross Validation` τις σωστές τιμές για την αριθμό των folds και το score που θας μας επιστρέψει, χρησιμοποιήσαμε δύο επιπλεόν παραμέτρους που θα παρέχουμε ως ορίσματα στην ρουτινά `cross_val_score`.
# 

# In[491]:


from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from itertools import product

class GridSearch():
    vthreshold = [0]
    n_components = [0]
    k_neighbors = [1]
    isknn = False
    
    def __init__(self, estimator, grid, cv='10', scoring='f1_macro', std_scale=True, over_sampling=True):
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        self.std_scale = std_scale
        self.over_sampling = over_sampling
        
        self.selector = VarianceThreshold()
        self.pca = PCA()
        self.scaler = StandardScaler()
        
        self.__set_grid__(grid)    
        
        self.X_train = None
        self.y_train = None
                
        if (str(estimator).split("(")[0] == "KNeighborsClassifier"):
            self.isknn = True
            
        self.best_estimator_ = None
        self.best_params_ = None
        
    def __set_grid__(self, grid):
        if not isinstance(grid, dict):
            raise("grid must be dict")
        
        for key, value in grid.items():
            if (key == "vthreshold"):
                self.vthreshold = sorted(value)
            elif (key == "n_components"):
                self.n_components = sorted(value)
            elif (key == "k_neighbors"):
                self.k_neighbors = sorted(value)
                
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        self.max_X_var = max(np.var(X_train, axis=0))
        
        cv_scores = {}
        
#         self.vthreshold = [i for i in self.vthreshold if i <= max_X_var]
        
#         self.n_components = [i for i in self.n_components if i <= X_train.shape[-1]]

        for vh, pca_n in product(self.vthreshold, self.n_components):
            self.selector.set_params(threshold=vh)
            
            X_train_mod = self.selector.fit_transform(self.X_train)
            X_test_mod = self.selector.transform(X_test)
            
            if (self.std_scale):
                X_train_mod = self.scaler.fit_transform(X_train_mod)
                X_test_mod = self.scaler.transform(X_test_mod)
            
            if (self.over_sampling):
                ros = RandomOverSampler(random_state=None, ratio=None, return_indices=False, sampling_strategy='auto')
                X_train_mod, y_train_mod = ros.fit_sample(X_train_mod, self.y_train)
            else:
                y_train_mod = self.y_train
    
            self.pca.set_params(n_components=pca_n)
            X_train_mod = self.pca.fit_transform(X_train_mod)
            
#             print("Variance Threshold: {0} PCA Components: {1}".format(vh, pca_n), end="")
            
            if self.isknn:
                for k in self.k_neighbors:
                    
                    if (k >= X_train_mod.shape[0]):
                        break
                    
                    self.estimator.set_params(n_neighbors=k)
                    
                    score = cross_val_score(self.estimator, X_train_mod, y_train_mod, cv=self.cv, scoring=self.scoring).mean()
                    cv_scores["VH:{0},PCA_N:{1},STD_SCL:{2},OVERSAMPL:{3},KNN:{4}".format(vh, pca_n, self.std_scale, self.over_sampling, k)] = score
                    
#                     print("\nKNN: {0} Score: {1}".format(k, score))
            else:                
                score = cross_val_score(self.estimator, X_train_mod, y_train_mod, cv=self.cv, scoring=self.scoring).mean()
                
                cv_scores["VH:{0},PCA_N:{1},STD_SCL:{2},OVERSAMPL:{3},KNN:NaN".format(vh, pca_n, self.std_scale, self.over_sampling)] = score
                
#                 print("\nScore: {0}".format(score))
             
        self.cv_scores = cv_scores
        self.__get_best_estimator_()
        
    def __fix_type(self, str_value):
        if (str_value == "True"):
            return True
        elif (str_value == "False"):
            return False
        elif (str_value == "NaN"):
            return -1
        else:
            return float(str_value)
        
    def __key2params(self, key_string):
        params_dict = {}
        splited_key = key_string.split(",")

        for attr in splited_key:
            key, value = attr.split(":")

            if (key == "VH"):
                params_dict["VarianceThreshold"] = self.__fix_type(value)
            elif (key == "PCA_N"):
                params_dict["PCA_components"] = self.__fix_type(value)
            elif (key == "STD_SCL"):
                params_dict["Standard_Scaler"] = self.__fix_type(value)
            elif (key == "OVERSAMPL"):
                params_dict["Over_Sampling"] = self.__fix_type(value)
            elif (key == "KNN"):
                params_dict["KNN_neighbors"] = self.__fix_type(value)

        return params_dict
    
    def __get_best_params_(self):
        max_score = -1
        max_score_params = ""

        for key,value in self.cv_scores.items():
            if value > max_score:
                max_score = value
                max_score_params = key

        self.max_score = max_score
        self.max_score_params = max_score_params
    
    def __get_best_estimator_(self):
        self.__get_best_params_()
        
        X_train_mod = self.X_train
        y_train_mod = self.y_train
        
        self.best_params_ = self.__key2params(self.max_score_params)
        
        # Variance Selector
        self.selector.set_params(threshold=self.best_params_["VarianceThreshold"])
        X_train_mod = self.selector.fit_transform(self.X_train)
        
        # Standard Scaler
        
        if (self.best_params_["Standard_Scaler"]):
            X_train_mod = self.scaler.fit_transform(X_train_mod)
        
        if (self.best_params_["Over_Sampling"]):
            ros = RandomOverSampler(random_state=None, ratio=None, return_indices=False, sampling_strategy='auto')
            X_train_mod, y_train_mod = ros.fit_sample(X_train_mod, y_train_mod)
        
        self.pca.set_params(n_components=int(self.best_params_["PCA_components"]))
        X_train_mod = self.pca.fit_transform(X_train_mod)
        
        if (self.isknn):
            self.estimator.set_params(n_neighbors = int(self.best_params_["KNN_neighbors"]))
            
        self.best_estimator_ = self.estimator
        
        self.X_train_mod = X_train_mod
        self.y_train_mod = y_train_mod
        
        self.best_estimator_.fit(X_train_mod, y_train_mod)
        
    def predict(self, X_test):
        X_test_mod = self.selector.transform(X_test)
        
        if (self.best_params_["Standard_Scaler"]):
            X_test_mod = self.scaler.transform(X_test)

        X_test_mod = self.pca.transform(X_test_mod)
        
        return self.best_estimator_.predict(X_test_mod)
     
    def __repr__(self):
        return str(self.estimator)
        


# Προσπαθήσαμε να κάνουμε τον χειρισμό όσο πιο όμοιο με τον `GridSearchCV` μιας και αυτόν διδαχτήκαμε και χρησιμοποιήσαμε στο εργαστήριο. Το αντικείμενο της `GridSearch` συμπεριφέρεται επίσης σαν έναν `συνολικό estimator`, που fitάρει στα δεδομένα εκπαίδευση και καταλήγει στον καλύτερο εκτιμητή που θα χρησιμοποιηθεί έπειτα σε μια ενδεχόμενη κλήση της `predict`. Ένα παράδειγμα παρουσιάζεται παρακάτω.

# In[370]:


grid = {
    "vthreshold" : np.arange(0, 0.06, 6),
    "n_components": [10, 11, 12, 15, 16, 17, 20, 21, 22],
    "k_neighbors" : [1, 3, 5, 7, 9]
}

knn = KNeighborsClassifier()
gs = GridSearch(knn, grid, cv=10, scoring="f1_micro", over_sampling=True, std_scale=True)

gs.fit(X_train, y_train)


# In[371]:


print(gs.best_params_)
print(gs.best_estimator_)


# In[372]:


preds_example = gs.predict(X_test)

print(classification_report(y_test, preds_example))


# ## Μέρος Δ (Συνέχεια)

# Με την χρήση λοιπόν της παραπάνω κλάσεις θα προσπαθήσουμε να βελτιστοποιήσουμε της υπερπαραμέτρους των ταξινομητών που αναφέραμε αρχικά. Στην περίπτωση των `Dummies` οι υπερπάραμετροι θα απευθύνονται αποκλειστικά στο πρώτο μέρος της αλυσίδας του pipeline αφού οι τελευταίοι δεν διαθέτουν παραμέτρους που μπορούμε να τροποποιήσουμε. Η ανάλυση της συμπεριφοράς τους έχει γίνει ήδη στο πρόηγουμενο μέρος, οπότε προς το παρόν θα αρκεστούμε κυρίως στην απόδοση τους. Απο την άλλη ο `KNeighbors` θα βελτιστοποιησή με βάση των αριθμών των γειτόνων και το είδος απόστασης που χρησιμοποιεί.  
# 
# Αξίζει σε αυτό το σημείο να υπολογίσουμε την μεγιστή διασπορά των δεδομένων καθώς είναι αυτή που θα μας οδηγήσει στις τιμές που θα πρέπει να ελέγχουμε για το `VarianceThreshold`
# 
# 
# Σε κάθε ταξινομήτή θα ελέγχουμε ένα γενικότερο εύρος τιμών, ενώ στην συνέχεια θα εκτελούμε μια πιο `progressive` αναζήτηση, σε πιο στενό εύρος. Εύκολα θα παρατηρήσουμε ότι το `VarianceThreshold` λόγω της μορφής των features, προτιμάται σχεδόν πάντα να έχει μηδενική τιμή.

# In[373]:


max(np.var(X_train, axis=0))


# In[374]:


import time 

dc_uniform = DummyClassifier(strategy="uniform")
dc_constant_0 = DummyClassifier(strategy="constant", constant=0)
dc_constant_1 = DummyClassifier(strategy="constant", constant=1)
dc_most_frequent = DummyClassifier(strategy="most_frequent")
dc_stratified = DummyClassifier(strategy="stratified")


# ### Dummy Uniform

# In[375]:


grid = {
    "vthreshold" : np.arange(0, 0.06, 6),
    "n_components": [10, 20, 30, 40],
}

gs_uni = GridSearch(dc_uniform, grid, cv=10, scoring="f1_micro", over_sampling=True, std_scale=True)

start_time = time.time()
gs_uni.fit(X_train, y_train)
preds_uni = gs_uni.predict(X_test)
print("Total time for fit and predict: %s seconds" % (time.time() - start_time))


# In[376]:


print(gs_uni.best_params_, gs_uni.best_estimator_)


# In[377]:


print(classification_report(y_test, preds_uni))


# Θα συνεχίσουμε την αναζήτηση μας κοντά στην περιοχή των `40 PCA Components` αν και οι προσδοκιές μας δεν είναι και πολύ υψηλές μιας και γνωρίζουμε την τυχαία συμπεριφορά του dummy uniform. 

# In[378]:


grid = {
    "vthreshold" : np.arange(0, 0.06, 6),
    "n_components": [38, 39, 40, 41, 42],
}

gs_uni = GridSearch(dc_uniform, grid, cv=10, scoring="f1_micro", over_sampling=True, std_scale=True)

start_time = time.time()
gs_uni.fit(X_train, y_train)
preds_uni = gs_uni.predict(X_test)
print("Total time for fit and predict: %s seconds" % (time.time() - start_time))


# In[379]:


print(gs_uni.best_params_, gs_uni.best_estimator_)


# In[380]:


print(classification_report(y_test, preds_uni))


# Κάθε φορά που τρέχουμε τον uniform λαμβάνουμε και διαφορετικά αποτέλεσμα, "λογικό" ως προς την τακτική που ακολουθεί, αδιάφορο όμως για ταξινόμηση.

# ### Dummy Constant 0

# In[381]:


grid = {
    "vthreshold" : np.arange(0, 0.06, 6),
    "n_components": [10, 20, 30, 40],
}

gs_con0 = GridSearch(dc_constant_0, grid, cv=10, scoring="f1_micro", over_sampling=True, std_scale=True)

start_time = time.time()
gs_con0.fit(X_train, y_train)
preds_con0 = gs_con0.predict(X_test)
print("Total time for fit and predict: %s seconds" % (time.time() - start_time))


# In[382]:


print(gs_con0.best_params_, gs_con0.best_estimator_)


# In[383]:


print(classification_report(y_test, preds_con0))


# In[384]:


grid = {
    "vthreshold" : np.arange(0, 0.06, 6),
    "n_components": [3, 4, 5, 6, 7, 8, 9],
}

gs_con0 = GridSearch(dc_constant_0, grid, cv=10, scoring="f1_micro", over_sampling=True, std_scale=True)

start_time = time.time()
gs_con0.fit(X_train, y_train)
preds_con0 = gs_con0.predict(X_test)
print("Total time for fit and predict: %s seconds" % (time.time() - start_time))


# In[385]:


print(gs_con0.best_params_, gs_con0.best_estimator_)


# In[386]:


print(classification_report(y_test, preds_con0))


# Όσα μικραίνουμε τα όρια του `PCA n_components` τόσο μικρότερο 'ζητάει' ο dummy constant, μιας και η συμπεριφορά του είναι ανεξάρτητη απο τα δείγματα. Παρόμοια συμπεριφορά περιμένουμε να δούμε στον constant 1.

# ### Dummy Constant 1

# In[387]:


grid = {
    "vthreshold" : np.arange(0, 0.06, 6),
    "n_components": [10, 20, 30, 40],
}

gs_con1 = GridSearch(dc_constant_1, grid, cv=10, scoring="f1_micro", over_sampling=True, std_scale=True)

start_time = time.time()
gs_con1.fit(X_train, y_train)
preds_con1 = gs_con1.predict(X_test)
print("Total time for fit and predict: %s seconds" % (time.time() - start_time))


# In[388]:


print(gs_con1.best_params_, gs_con1.best_estimator_)


# In[389]:


print(classification_report(y_test, preds_con1))


# Ότι ακριβώς περιμέναμε συνέβη. Πάλι ο σταθερός ταξινόμητής φαίνεται να 'ζητάει' όσο το δυνατόν λιγότεες συνιστώσεις στον `PCA`. Επειδή ξέρουμε ότι η ακρίβεια δεν πρόκειται να αλλάξει μιας και μιλάμε για `σταθερό` ταξινομητή προχωράμε στον επόμενο.

# ### Dummy Μost Frequent

# In[390]:


grid = {
    "vthreshold" : np.arange(0, 0.06, 6),
    "n_components": [3, 4, 5, 6, 7, 8, 9],
}

gs_mf = GridSearch(dc_most_frequent, grid, cv=10, scoring="f1_micro", over_sampling=True, std_scale=True)

start_time = time.time()
gs_mf.fit(X_train, y_train)
preds_mf = gs_mf.predict(X_test)
print("Total time for fit and predict: %s seconds" % (time.time() - start_time))


# In[391]:


print(gs_con0.best_params_, gs_mf.best_estimator_)


# In[392]:


print(classification_report(y_test, preds_mf))


# Θα συνεχίσουμε την αναζήτηση μας κοντά στην περιοχή των `1-5 PCA Components` αν και οι προσδοκιές μας δεν είναι και πολύ υψηλές μιας και γνωρίζουμε ότι διαλέγει με βάση την most frequent κλάση, δηλαδή την κλάση 0, όπως ο αντίστοιχος constant.

# In[393]:


grid = {
    "vthreshold" : np.arange(0, 0.06, 6),
    "n_components": [0, 1, 2, 3, 4],
}

gs_mf = GridSearch(dc_most_frequent, grid, cv=10, scoring="f1_micro", over_sampling=True, std_scale=True)

start_time = time.time()
gs_mf.fit(X_train, y_train)
preds_mf = gs_mf.predict(X_test)
print("Total time for fit and predict: %s seconds" % (time.time() - start_time))


# In[394]:


print(gs_con0.best_params_, gs_mf.best_estimator_)


# In[395]:


print(classification_report(y_test, preds_mf))


# ### Dummy Stratified

# In[396]:


grid = {
    "vthreshold" : np.arange(0, 0.06, 6),
    "n_components": [10, 20, 30, 40],
}

gs_st = GridSearch(dc_stratified, grid, cv=10, scoring="f1_micro", over_sampling=True, std_scale=True)

start_time = time.time()
gs_st.fit(X_train, y_train)
preds_st = gs_st.predict(X_test)
print("Total time for fit and predict: %s seconds" % (time.time() - start_time))


# In[397]:


print(gs_st.best_params_, gs_st.best_estimator_)


# In[398]:


print(classification_report(y_test, preds_st))


# In[399]:


grid = {
    "vthreshold" : np.arange(0, 0.06, 6),
    "n_components": [28, 29, 30, 31, 32, 33],
}

gs_st = GridSearch(dc_stratified, grid, cv=10, scoring="f1_micro", over_sampling=True, std_scale=True)

start_time = time.time()
gs_st.fit(X_train, y_train)
preds_st = gs_st.predict(X_test)
print("Total time for fit and predict: %s seconds" % (time.time() - start_time))


# In[400]:


print(gs_st.best_params_, gs_st.best_estimator_)


# In[401]:


print(classification_report(y_test, preds_st))


# Στα υπόλοιπα ερωτήματα θα απαντήσουμε συγκεντρωτικά για όλους του `Dummy` και ξεχωριστά για τον `KNN`. Αρχικά θα παρουσίασουμε τα `bar plots`, στα οποία, όποιες διαφορές και να δούμε συγκριτικά με αυτά που είδαμε στο Μέρος Γ, οφείλονται στην τυχαίοτητα ορισμένων εξ αυτών, και συγκεκριμένα `uniform`,`stratified`.

# In[402]:


gs_dm_clfs = [gs_uni, gs_con0, gs_con1, gs_mf, gs_st]

barplot_score(gs_dm_clfs, X_test, y_test)


# Επείδη ο συμπεριφορά των dummies δεν επηρέαζεται κατά κύριο λόγο, απο την μεταβολή των υπερπαραμέτρων του πρώτου σταδίο, η ανάλυση που έχει πραγματοποιηθεί ήδη στο Μέρος Γ, μπορεί πολύ εύκολα να επαναληφθεί και εδώ. Όσο για τον χρόνο επειδή οι τιμές που μετρήσαμε είναι αρκετά μικρές, καθώς δεν 'κουράζονται' στην προβλέψη τους, δεν κρίναμε αναγκαίο να τους τυπώσουμε με μορφή πίνακα.

# In[404]:


plot_report(gs_dm_clfs, X_test, y_test, ["Rock", "Metal"])


# ### ΚΝΝ Μέρος Δ
# Στο δεύτερο μισό του Μέρους Δ, όπως δηλώνει και ο τίτλος, θα αφιερωθούμε αποκλειστικά στον `KNN` καθώς είναι ο μόνος απο τους ταξινομητές που επιδέχεται παραμετροποιήση που αντανακλά στα αποτελέσματα. Θα χρησιμοποιήσουμε την κλάση `GridSearch` που φτιάξαμε για να προσεγγίσουμε καλές τιμές για τις υπερπαραμέτρους, και έπειτα θα προχωρήσουμε σε πιο ενδελεχή χειροκίνητο έλεγχο. Θα ξεκινήσουμε με μετρική `micro` και επειτα θα εξετάσουμε και την `macro`.

# In[458]:


grid = {
    "vthreshold" : np.arange(0, 0.06, 6),
    "n_components": [5, 10, 15, 20, 25, 30, 35, 40, 45],
    "k_neighbors" : [1, 3, 5, 7, 9, 11, 12, 15]
}

knn_s = KNeighborsClassifier()
gs_knn_s = GridSearch(knn_s, grid, cv=10, scoring="f1_micro", over_sampling=True, std_scale=True)

start_time = time.time()
gs_knn_s.fit(X_train, y_train)
preds_knn_s = gs_knn_s.predict(X_test)
print("Total time for fit and predict: %s seconds" % (time.time() - start_time))


# In[459]:


print(gs_knn_s.best_params_, gs_knn_s.best_estimator_)


# In[460]:


print(classification_report(y_test, preds_knn_s))


# Οι τιμές που επιλέξαμε για το `VarianceThreshold` και πάλι δεν συμβάλλουν στην τελική επιλογή, καθώς όπως έχουμε αναφέρει πολλάκις, εφόσον η μέγιστη διασπορά που συναντάμε είναι της τάξης 0.06, προκύπτει ότι τα δεδομένα δεν χρειάζονται τέτοιου είδους προπεξεργασία. Για αυτό θα κρατήσουμε το `VarianceThreshold` σταθερό στο μηδέν και θα ασχοληθούμε με τις υπόλοιπες υπερπαραμέτρους. 

# Ως πρώτη αλλαγή θα αφαιρέσουμε το over sampling για να δούμε πως θα αντιδράσει το `GridSearch` ως προς την επιλογή των υπόλοιπων παραμέτρων.

# In[450]:


grid = {
    "vthreshold" : [0],
    "n_components": [5, 10, 15, 20, 25, 30, 35, 40, 45],
    "k_neighbors" : [1, 3, 5, 7, 9, 11, 12, 15]
}

knn_s1 = KNeighborsClassifier()
gs_knn_s1 = GridSearch(knn_s1, grid, cv=10, scoring="f1_micro", over_sampling=False, std_scale=True)

start_time = time.time()
gs_knn_s1.fit(X_train, y_train)
preds_knn_s1 = gs_knn_s1.predict(X_test)
print("Total time for fit and predict: %s seconds" % (time.time() - start_time))


# In[451]:


print(gs_knn_s1.best_params_, gs_knn_s1.best_estimator_)


# In[452]:


print(classification_report(y_test, preds_knn_s1))


# Και τα δύο αποτελέσματα αν και σχετικά ίσα, μας υποδηλώνουν δύο πράγματα.
# - Οι γείτονες του `KNN` που οδηγούν στην μεγαλύτερη τιμή του $f1$ κυμαίνεται είναι μία εκ των τριών $[1,3,5]$, και κυρίως $[1,3]$.
# - Οι τιμές μεταξύ $[10, 25]$ εμφανίζουν καλύτερα αποτελέσματα για τα PCA Components

# Σύμφωνα με τις παραπάνω δύο παρατηρήσεις θα περιορίσουμε την περιοχή των τιμών γύρω απο αυτές που είχαν υψηλότερη απόδοση.

# In[492]:


grid = {
    "vthreshold" : [0],
    "n_components": [10, 15, 18, 19, 20, 21, 22],
    "k_neighbors" : [1, 3, 5]
}

knn_s3 = KNeighborsClassifier()
gs_knn_s3 = GridSearch(knn_s1, grid, cv=10, scoring="f1_micro", over_sampling=False, std_scale=True)

start_time = time.time()
gs_knn_s3.fit(X_train, y_train)
preds_knn_s3 = gs_knn_s3.predict(X_test)
print("Total time for fit and predict: %s seconds" % (time.time() - start_time))


# In[493]:


print(gs_knn_s3.best_params_, gs_knn_s3.best_estimator_)


# In[494]:


print(classification_report(y_test, preds_knn_s3))


# Οι υπεράμετροι που επιλέγει είναι οι ίδιοι με προηγούμενως. Σε αυτό το σημείο θα ασχοληθούμε με την χειροκίνητη παραμετροποίηση, λίγων τιμών.
# 
# - k=1, n=20
# - k=1, n=21

# In[507]:


selector = VarianceThreshold()
scaler = StandardScaler()
pca = PCA(n_components=20)
clf = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)

X_train_mod = scaler.fit_transform(X_train)
X_train_mod = pca.fit_transform(X_train_mod)

clf.fit(X_train_mod, y_train)

X_test_mod = scaler.transform(X_test)
X_test_mod = pca.transform(X_test_mod)

preds = clf.predict(X_test_mod)

print(precision_recall_fscore_support(y_test, preds, average="micro"))
print(classification_report(y_test, preds))


# In[508]:


selector = VarianceThreshold()
scaler = StandardScaler()
pca = PCA(n_components=21)
clf = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)

X_train_mod = scaler.fit_transform(X_train)
X_train_mod = pca.fit_transform(X_train_mod)

clf.fit(X_train_mod, y_train)

X_test_mod = scaler.transform(X_test)
X_test_mod = pca.transform(X_test_mod)

preds = clf.predict(X_test_mod)

print(precision_recall_fscore_support(y_test, preds, average="micro"))
print(classification_report(y_test, preds))


# Τα αποτέλεσμα δεν διαφέρουν καθόλου για αυτό θα παραμείνουμε στην τιμή 20 για την παράμετρο `PCA Components`, και θα δοκιμάσουμε μεταξύ δύο τιμών για τον `KNN`.
# 
# - k=1
# - k=3

# In[509]:


selector = VarianceThreshold()
scaler = StandardScaler()
pca = PCA(n_components=20)
clf = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)

X_train_mod = scaler.fit_transform(X_train)
X_train_mod = pca.fit_transform(X_train_mod)

clf.fit(X_train_mod, y_train)

X_test_mod = scaler.transform(X_test)
X_test_mod = pca.transform(X_test_mod)

preds = clf.predict(X_test_mod)

print(precision_recall_fscore_support(y_test, preds, average="micro"))
print(classification_report(y_test, preds))


# Εν τέλει το αποτέλεσμα του `GridSearch` μας δικαιώνει, καθώς οι βέλτιστες τιμές είναι αυτές που μας επέστρεψε εξ αρχής. Η απόκλιση που υπάρχει ευθύνεται αποκλειστικά στο `VarianceThreshold` που αφαιρεί στήλες με μηδενική διασπορά. Στην συνέχεια θα παρουσιάσουμε τους πίνακες σύγχυσης αλλά και τα barplots των μεταβολών μεταξύ των ταξινομητών που δοκιμάσαμε.

# In[521]:


gs_knns = [gs_knn_s, gs_knn_s1]

barplot_score(gs_knns, X_test, y_test)


# In[522]:


barplot_score(clf, X_test_mod, y_test)


# In[523]:


plot_report(gs_knns, X_test, y_test, ["Rock", "Metal"])


# Τα τελικά αποτέλεσμα είναι αυτά ακριβώς που περιγράψαμε και παραπάνω. Λόγω της `GridSearch` δεν χρειάστηκε να κάνουμε εξαντλητική αναζήτηση χειροκίνητα πράγμα που μας εξασφάλισε χρόνο και ενέργεια. Ωστόσο ένα σημείο που αξίζει μια παραπάνω προσοχή, είναι το γιατί ένας τόσος μικρός αριθμός απο `PCA Components` οδηγεί στα βέλτιστα αποτελέσματα. Σε αυτό θα μας βοηθήσει αν τυπωσουμε το συσσωρευτικό ποσοστό διασποράς για `n=25`

# In[526]:


pca = PCA(n_components=50)
pca.fit_transform(X_train)

evar = pca.explained_variance_ratio_
cum_evar = np.cumsum(evar)
print(cum_evar)
plt.figure(1, figsize=(5, 5))
plt.xlabel("Principal Component number")
plt.ylabel('Cumulative Variance')
plt.plot(cum_evar, linewidth=2)
plt.show()


# Παρατηρούμε ότι για τιμές 20+ οι κύριες συνιστώσες καταφέρνουν να περιγράψουν σχεδον το $95%$ των δεδομένων μας. Για αυτό το λόγω όταν οι διαστάσει ξεφέυγουν πάνω απο 25+, τα δεδομένα μας δεν αρκούν για να εκπαιδεύσουν πλήρως τον ταξινομητή μας. (Curse of dimensionality)

# In[ ]:




