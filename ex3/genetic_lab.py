#!/usr/bin/env python
# coding: utf-8

# # Άσκηση 3: Βελτιστοποίηση συναρτήσεων με Γενετικούς Αλγόριθμους
# ### Παπακωνσταντίνου Πολύβιος 03114892
# ### Πατρής Νικόλαος 03114861
# ##### Ομάδα Α30
# 
# <img src="http://infinity77.net/global_optimization/_images/Alpine01.png" alt="Alpine01" style="width: 500px;"/>
# 
# Στόχος της άσκησης είναι η βελτιστοποίηση συναρτήσεων χωρίς παραγώγους (derivative free optimization) με χρήση Γενετικών Αλγόριθμων μέσω της βιβλιοθήκης DEAP. Η βελτιστοποίηση χωρίς παραγώγους είναι ιδιαίτερα χρήσιμη σε περιπτώσεις όπου η αντικειμενική συνάρτηση $f$ δεν είναι διαθέσιμη (black-box optimization) ή σε περίπτωσεις που είναι, είναι δύσκολο ή μη πρακτικό να υπολογιστούν οι παράγωγοί της. Για παράδειγμα η  $f$ μπορεί να μην είναι διαφορίσιμη, η παραγώγιση της μπορεί να είναι δύσκολη ή να απαιτεί πολύ χρόνο,  η $f$ να περιέχει θόρυβο έτσι ώστε οι μέθοδοι που βασίζονται σε απειροστικές διαφορές να μην είναι αποτελεσματικές ή να υπάρχουν πολλά τοπικά βέλτιστα μεταξύ άλλων. 

# 
# ### Βοηθητική συνάρτηση εκτύπωσης υπερπαραμέτρων και απόδοσης (σχετικής και απόλυτης)
# 
# Φτιάχνουμε τέλος μια βοηθητική συνάρτηση που τυπώνει συγκεντρωτικά τα αποτελέσματα ως εξής (η επεξήγηση για τις δύο πρώτες κολόνες στο ακριβώς επόμενο section):
# 
# | operators                      | strategy                      | successes | s.avg.min | s.avg.evals | s.avg.gens | avg.evals | avg.min   | avg.time |
# |--------------------------------|-------------------------------|-----------|-----------|-------------|------------|-----------|-----------|----------|
# |CrossoverLow,MutationLow,SelectionLow | eaSimple 200 0.8 0.2          | 1         | 8.508e-42 | 16487       | 96         | 17014     | 9.115e-34 | 1.898    |
# | CrossoverHigh,MutationLow,SelectionLow | eaMuPlusLambda 50 150 0.8 0.2 | 1         | 3.885e-41 | 14750       | 98         | 15050     | 1.631e-21 | 1.591    
# 
# Ευτυχώς δεν χρειάστηκε να σκεφτούμε περίπλοκες υλοποιήσεις καθώς το DataFrame που παράγει η βιβλιοθήκη panda είναι ακριβώς αυτό που χρειαζόμαστε.

# In[1]:


from deap import algorithms
from deap import base, creator, tools
import pandas as pd
import numpy 
import time
import itertools 

pd.set_option('display.max_colwidth', -1)

def report(data, columns_names=None):
    frame = pd.DataFrame(data, columns=columns_names)
    return frame

data = [['TestTestTest', 2], [3, "GGG"]]
report(data, ["Foo", "Bar"])


# 
# ## Μέρος 1. Βελτιστοποίηση μη κλιμακούμενης συνάρτησης
# 
# Η μη-κλιμακούμενη συνάρτηση που αντιστοιχεί στον αριθμό της ομάδα μας είναι η  `Corana [37]`. Ύστερα απο συζήτηση που είχαμε με τον κύριο Σιόλα μας πρότεινε να χρησιμοποιήσουμε την συγκεκριμένη υλοποίηση, καθώς προβλέπει και για κάποιες ασάφειες που υπάρχουν στον 

# In[3]:


import math as mt
 
def corana(x):
    # Credits: Giorgos Siolas
    # https://al-roomi.org/benchmarks/unconstrained/4-dimensions/90-corana-s-function
    # but also added check for z[i]!=0 from https://github.com/estsauver/yolo-nemesis/blob/master/coranaEval.m
    
    s=0.2
    d=[1,1000,10,100,10]
    z=[]
    sum=0
    for i in range(0, len(x)):
        z.append(0.2*mt.floor(mt.fabs(x[i]/s)+0.49999)*mt.copysign(1, x[i]))
        if mt.fabs(x[i]-z[i]) < 0.05 and z[i]!=0:
            sum = sum + 0.15*d[i]*(z[i]-0.05*mt.copysign(1, z[i]))**2
        else:
            sum = sum + d[i]*x[i]**2    
    return sum


# In[4]:


corana([0,0,0,0])


# 
# ### Εύρεση βέλτιστου συνδυασμού τελεστών - στρατηγικής
# 

# 
# #### Γενετικοί τελεστές
# 
# Αρχικά θα δοκιμάσουμε δύο διαφορετικούς τελεστές διασταύρωσης και δύο διαφορετικούς τελεστές μετάλλαξης της επιλογής μας, με δύο διαφορετικές τιμές υπερ-παραμέτρων για τον καθένα. Θα δοκιμάσουμε επίσης δύο τιμές υπερ-παραμέτρων για τον τελεστή επιλογής selTournament. Έχουμε δηλαδή συνολικά 32 συνδυασμούς τελεστών. Σε αυτά αντιστοιχούν οι dummy ονομασίες της κολόνας "operators" (πλην του selTournament).

# ### Crossover

# ##### cxBlend

# Ως πρώτο βήμα θα χρειαστεί να επιλέξουμε τους τελεστές διαστάυρωσης που θα χρησιμοποιήσουμε στον γενετικό μας αλγόριθμο. Οι τελεστές θα πρέπει να είναι σε θέση να διαχειρίζονται ίδιο τύπο αριθμών/γονιδίων ώστε να διασταυρώνουν σωστά τα χρωμοσώματα των ατόμων (individual). 
# 
# Εφόσον έχουμε πραγματικούς αριθμούς θα χρησιμοποιήσουμε αρχικά τον `cxBlend` που ανακατεύει το γενετικό υλικό των γονέων $x_1$ και $x_2$ σε κάθε διάσταση $i$ με τυχαίο τρόπο και ανάλογο της παραμέτρου $\alpha$: 
# 
# $\gamma = (1 + 2 \cdot \alpha) \cdot  random() - \alpha\\
# ind1[i] = (1 - gamma) \cdot x_1[i] + gamma \cdot x_2[i]\\
# ind2[i] = gamma \cdot x_1[i] + (1 - gamma) \cdot x_2[i]$
# 
# Οι τιμές που θα δοκιμάσουμε για την υπερπαράμετρο `alpha` θα πρέπει να είναι ο συνδυασμός μιας υψηλής που θα οδηγεί σε μεγαλύτερη μεταβολή του γενετικού υλικού και μιας μικρή για τον προφανή λόγο. Μπορούμε να κατασκευάσουμε μία συνάρτηση για να παρατηρήσουμε τις μεταβολές που επιβάλλει στην `gamma` η υπερπαράμετρος `alpha`. Θα τρέξουμε ένα αριθμό γύρων και θα δοκιμάζουμε 50 τιμές στο διάστημα 1-20. Μετά την ολοκλήρωση των γύρων θα υπολογίσουμε την μέση τιμή της `gamma` για κάθε μία απο τις 50 τιμές. Δυστυχώς δεν μπορούμε να προβλέψουμε την τιμή της `gamma` στο διάστημα 0-1, ώστε να καθορίσουμε ποιου ατόμου το χρωμόσωμα θα κυριαρχήσει.

# In[5]:


import random
import numpy as np

gamma = lambda alpha: (1+2*alpha) * random.random() - alpha

def gamma_mean(low=1, high=20, num=50, rounds=100):
    alpha_values = np.linspace(low, high, num)

    gamma_mean = []
    for i in range(rounds):
        gammas = [gamma(j) for j in alpha_values]
        gamma_mean.append(gammas)

    return alpha_values, np.mean(gamma_mean,0)


# In[7]:


import matplotlib.pyplot as plt

alphas, gammas = gamma_mean()

plt.plot(alphas, gammas, 'ro')
plt.show()


# Παρατηρούμε ότι με τιμές κοντά στο 1, η υπεπαράμετρος `alpha` 
# αναμειγνύει καλύτερα τα γονίδια των δύο ατόμων καθώς φαίνεται να ισομοιράζει τις τιμές σε κάθε θέση $(gamma \sim 0.5)$. Όσο οι τιμές μεγαλώνουν και η `gamma` ξεπερνά το 1, δηλαδή η `alpha` γίνεται μεγαλύτερη απο το 5, οι αλλαγές στα γονίδια γίνονται όλο και πιο έντονες. Ως εκ τούτου οι δύο τιμές που θα επιλέξουμε θα είναι: $1,5$.

# ##### cxSimulatedBinary

# O δεύτερος τελεστής που επιλέξαμε θα είναι ο `cxSimulatedBinary`. Όπως και με τον προηγούμενο, δέχεται μία υπεπαράμετρο, την `eta`. Η παράμετρος αυτή επηρεάζει της διαδικασία της διαστάυρωσης των γονιδίων με τον εξής τρόπο.
# 
# ```python
# if rand <= 0.5:
#     beta = 2. * rand
# else:
#     beta = 1. / (2. * (1. - rand))
# beta **= 1. / (eta + 1.)
# 
# ind1[i]=0.5*(((1 + beta) * x1) + ((1 - beta) * x2))
# ind2[i]=0.5*(((1 - beta) * x1) + ((1 + beta) * x2))
# ```
# 
# Θα ακολουθήσουμε παρόμοια διαδικασία με το προηγούμενο ώστε να ανιχνεύσουμε τις μεταβολές της `eta` στα γονίδια των ατόμων. Η μεταβλητή που ευθύνεται για τις μεταβολές είναι η `beta`, για αυτό το λόγο θα είναι και αυτή που θα οπτικοποιήσουμε.

# In[8]:


import random
import numpy as np

def beta(eta):
    rand = random.random()
    if rand <= 0.5:
        beta = 2. * rand
    else:
        beta = 1. / (2. * (1. - rand))
    beta **= 1. / (eta + 1.)
    
    return beta

def beta_mean(low=1, high= 50, num=50, rounds=100):
    eta_values = np.linspace(low, high, num)

    beta_mean = []
    for i in range(rounds):
        betas = [beta(j) for j in eta_values]
        beta_mean.append(betas)

    return eta_values, np.mean(beta_mean,0)


# In[9]:


import matplotlib.pyplot as plt

etas, betas = beta_mean()

plt.plot(etas, betas, 'ro')
plt.show()


# Όπως παρατηρούμε καθώς η τιμή της `eta` μεγαλώνει η τιμή της `beta` συγκλίνει στο 1, δημιουργώντας μια στατική διασταύρωση (stationary crossover) των γονιδίων, με αποτέλεσμα οι τιμές των παδιών να μην αποκλίνουν με τις τιμές των γονέων. Εμείς ενδιαφερόμαστε για τιμές της `eta` που δημιουργούν contracting ή expanding crossovers. Για αυτό θα περιορίσουμε το εύρος αναζήτησης στο διάστημα 0-1.

# In[10]:


etas, betas = beta_mean(low=0, high=1)

plt.plot(etas, betas, 'ro')
plt.show()


# Όσο πιο μικρή τιμή δίνουμε στην υπεπαράμετρο `eta` τόσο μεγαλύτερη αλλαγή δημιουργεί στο γενετικό υλικό των ατόμων. Σύμφωνα λοιπόν με τα παραπάνω δύο αντιπροσεπευτικές τιμές είναι: $0.2, 0.8$

# ### Mutation

# Αφού λοιπόν ολοκληρώσαμε το πρώτο βήμα της επιλογής των τελεστών διασταύρωσης, περνάμε στο δεύτερο που είναι η επιλογή τελεστών μετάλλαξης μαζί με τις δύο ακραίες τιμές (high, low) των υπερπαραμέτρων τους.

# #### mutGaussian

# Έχοντας πλέον εξοικειωθεί με την γκαουσιανή μετάλλαξη ήδη από τα εργαστήρια και γνωρίζοντας βεβαίως την λειτουργικότητας με floating αριθμούς θα είναι η πρώτη μας επιλογή ως τελεστή. Η συνάρτηση αυτή χαρακτηρίζεται από ένα ζεύγος υπερπαραμέτρων: της μέσης τιμής και της διασποράς, οι οποίες λειτουργούν με τον εξής τρόπο.
# 
# ```python
# individual[i] += random.gauss(m, s)
# ```
# Η παραπάνω εντολή είναι εκείνη που περιγράφει την λειτουργία του συγκεκριμένου τελεστή μετάλλαξης, προσθέτει απλά γκαουσιανό θόρυβο σε κάθε γονίδιο του ατόμου. Είναι προφανές λοιπόν ότι η υπερπαράμετρος που καθορίζει την επίδραση που θα έχει η μετάλλαξη σε κάθε άτομο είναι κυρίως η μέση τιμή, καθώς αν επιλέξουμε μία μικρή τιμή για την διασπορά ο θόρυβος που θα προστίθεται θα έχει τιμές κοντά στην μέση τιμή. Δύο καλές τιμές θα ήταν $(0,1), (20,10)$. Η τιμή της `indpb`, δηλαδή της πιθανότητας να τροποποιηθεί κάθε γονίδιο του χρωμοσώματος θα την έχουμε σταθερή στο 0.5, ώστε να μην μεροληπτούμε υπέρ ή κατά της μετάλλαξης.

# #### mutPolynomialBounded

# Η δεύτερη επιλογή μας για τελεστή μετάλλαξης είναι μονόδρομος \*\*. Δυστυχώς λόγω του περιορισμένου αριθμού των τελεστών μετάλλαξης, ο μόνος εκτός του `mutGaussian` που επιτρέπεται για την χρήση `float` αριθμών είναι ο `mutPolynomialBounded`. Οι υπερπαράμετροι που χρησιμοποιεί ο τελεστής είναι οι εξής: $eta, low, up, indp$. Ήδη από όσα έχουμε ήδη περιγράψει η πρώτη και η τελευταία παράμετρος έχει αναλυθεί αρκετά ως προς τον τρόπο λειτουργίας της καθώς και για την επίδραση που έχει στα "γεννετικό υλικό" των ατόμων. Η υλοποίηση του συγκεκριμένου μηχανισμού παρούσιαζεται παρακάτω. Ο λόγος για την επιλογή μας, όπως και με τα προηγούμενα, είναι ότι θα μας βοηθήσει αρκετά στην επεξέγηση και στην αιτιολόγηση των επιλογών μας.
# 
# ```python
# if random.random() <= indpb:
#     x = individual[i]
#     delta_1 = (x - xl) / (xu - xl)
#     delta_2 = (xu - x) / (xu - xl)
#     rand = random.random()
#     mut_pow = 1.0 / (eta + 1.)
# 
#     if rand < 0.5:
#         xy = 1.0 - delta_1
#         val = 2.0 * rand + (1.0 - 2.0 * rand) * xy**(eta + 1)
#         delta_q = val**mut_pow - 1.0
#     else:
#         xy = 1.0 - delta_2
#         val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy**(eta + 1)
#         delta_q = 1.0 - val**mut_pow
# 
#     x = x + delta_q * (xu - xl)
#     x = min(max(x, xl), xu)
#     individual[i] = x
# ```
# 
# Αμέσως λοιπόν μπορούμε να παρατηρήσουμε την επίδραση μια αρκετά μεγάλης τιμής για την παράμετρο `eta`. Η τοπική μετάβλητη `mut_pow` συγκλίνει προς το μηδέν καθώς η τιμής της `eta` μεγαλώνει παράλληλα με την `delta_q`: 
# $$delta\_q = (1-val^{mut\_pow}) \Rightarrow^{\{mut\_pow \rightarrow 0\}}= (1-val^0) = 1-1=0$$. 
# Παρόμοια συμπεριφορά εμφανίζει και όταν τα `low`,`up` είναι αρκετά κοντά, καθώς τείνει να μηδενιστεί. Στην αντίθετη περίπτωση όπου οι δύο προηγούμενες παράμετροι αποκτούν μεγάλη απόκλιση μεταξύ τους η τροποποιήση που υφίσταται το κάθε γονίδιο είναι αρκετά μεγάλη. Αφού λοιπόν κατάφεραμε να ερμηνεύσουμε την επίδραση των παραμέτρων το μόνο που μένει είναι να βρούμε δύο αντιπροσωπευτικές τιμές. 
# 
# Ωστόσο υπάρχει ένα σημαντικό σημείο που πρέπει να προσέξουμε. Επειδή η `mut_pow` είναι πάντα μικρότερη του 1, ενδέχεται όταν την χρησιμοποιήσουμε για να υψώσουμε κάτι σε δύναμη (π.χ. την `val`) να προκύξει μιγαδικός αριθμός. Θα πρέπει επομένως να κρατήσουμε την `val` θετική. Ένας εύκολος τρόπος θα ήταν να κρατήσουμε τις μεταβλητές `delta_1, delta_2` θετικές (θα είναι επίσης και $<1$). Θα μπορούσαμε να θέσουμε τις τιμές των `low`, `high` τέτοιες ώστε να σχηματίζουν ένα διάστημα που περιλάμβανει τα τις επιτρεπές τιμές για τα γονίδια (-100,100). Θα παρατηρήσουμε όμως ότι δεν μπορούμε να είμαστε σίγουροι για τις νέες τιμές που θα προκύψουν απο τα στάδιο της διαστάυρωσης, με αποτέλεσμα να εμφανιστεί και πάλι αρνητικό υπόρριζο. Η **μόνη λύση** θα είναι να κατασκευάσουμε τον ίδιο τελεστή μετάλλαξης και στην περίπτωση που η `delta_q` είναι μιγαδικός αριθμός να χρησιμοποιούμε μόνο το πραγματικό μέρος του.

# \*\* Παραμένει μόνο η `mutShuffleIndexes()` αλλά δεν μας καλύπτει ένα απλό μπέρδεμα το indices ως προς το κομμάτι της μετάλλαξης.

# In[11]:


def mutPolynomialBounded(individual, eta, low, up, indpb):
    size = len(individual)
    low = [low] * size
    up = [up] * size

    for i, xl, xu in zip(range(size), low, up):
        x = individual[i]
        delta_1 = (x - xl) / (xu - xl)
        delta_2 = (xu - x) / (xu - xl)
        rand = random.random()
        mut_pow = 1.0 / (eta + 1.)

        if rand < 0.5:
            xy = 1.0 - delta_1
            val = 2.0 * rand + (1.0 - 2.0 * rand) * xy**(eta + 1)
            delta_q = val**mut_pow - 1.0
        else:
            xy = 1.0 - delta_2
            val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy**(eta + 1)
            delta_q = 1.0 - val**mut_pow

        if isinstance(delta_q, (complex)):
            delta_q = delta_q.real

        x = x + delta_q * (xu - xl)
        x = min(max(x, xl), xu)
        individual[i] = x

    return individual,


# Σύμφωνα με όσα είπαμε μια επιλογή για την `eta`θα ήταν λίγο μεγαλύτερη απο το 1, ώστε η επίδραση να μην είναι καταλυτική για τις νέες τιμές των γονιδίων. Η `inpd` θα μείνει και πάλι στο $0.5$ για τον ίδιο λόγο με τον προήγουμενο τελεστή μετάλλαξης, ενώ όσο για το `low`, `up` οι τιμές τους θα είναι αυτές που θα καθορίσουν την χαμηλή και υψηλή επίδραση που ζητείται στην εκφώνηση.
# 
# - eta = 0.2, low=-5, up=5, indpb=0.5
# - eta = 0.002, low=-100, up=100, indpb=0.5

# ### Selection

# #### selTournament

# Ο τελευταίος τελεστής υλοποιεί την διαδικάσια της φυσικής επιλογής του γενετικού αλγόριθμου. Παρόλο που στο **documentaion** φαίνεται να μπορούμε να ορίσουμε και δεύτερη παράμετρο εκτός απο το `tournsize`. H υλοποίηση όμως των στρατηγικών της εξέλιξης έχουν θέσει apriori την τιμή της `k` να είναι ίση με το μέγεθος του πληθυσμού. 
# 
# ```python
# --eaSimple--
# offspring = toolbox.select(population, len(population))
# ------------
# ```
# 
# ```python
# def selTournament(individuals, k, tournsize, fit_attr="fitness"):
#     chosen = []
#     for i in xrange(k):
#         aspirants = selRandom(individuals, tournsize)
#         chosen.append(max(aspirants, key=attrgetter(fit_attr)))
#     return chosen
# ```
# 
# Επομένως επιλέγονται κάθε φορά $k = len(population)$ άτομα μέσω της `selRandom`, η οποία διαλέγει τυχαία `tournsize` άτομα από τα οποία η `selTournament` επιλέγει αυτό με την μεγαλύτερη `fit_attr` τιμή. Αν λοιπόν μεγαλώσουμε ή μικρύνουμε αρκετά την τιμή του `tournsize` ενδεχομένως να χρειαστούν αρκετές γενιές για να βρεθεί η λύση. Σύμφωνα με όσα αναφέραμε οι δύο τιμές που θα επιλέξουμε για την υπεπαράμετρο `tournsize` είναι: $5,30$.

# 
# 
# 
# #### Στρατηγική εξέλιξης
# 
# Για κάθε συνδυασμό τελεστών θα εξετάσουμε τρεις στρατηγικές: τον **απλό ΓΑ** και τις στρατηγικές εξέλιξης **“μ+λ”** και **“μ,λ”**. Στην κολόνα strategy βλέπετε το όνομα της στρατηγικής και μετά την τιμή ή τις τιμές που χαρακτηρίζουν τον πληθυσμό. Προφανώς o απλός γενετικός έχει μόνο το μέγεθος του πληθυσμού ενώ οι δύο άλλες στρατηγικές έχουν τις τιμές για τα μ και λ. Οι δύο τελευταίοι αριθμοί της κολόνας αντιστοιχούν στις πιθανότητες διασταύρωσης και μετάλλαξης.
# 

# 
# #### Μεθοδολογία εύρεσης βέλτιστου συνδυασμού
# 
# Αρχικά θα προσδιορίσουμε τους καταλληλότερους συνδυασμούς τελεστών-στρατηγικών. Για το λόγο αυτό θα θέσουμε ένα σχετικά μικρό αριθμό γύρων (π.χ. 5-10) και μέγιστων γενεών (π.χ. 50-100-150) και σταθερές τιμές στις παραμέτρους πληθυσμού και τις πιθανότητες διασταύρωσης και μετάλλαξης. 
# 
# Θα διαλέξουμε ένα αρχικό `delta` κοντά στο ολικό ελάχιστο του καλύτερου συνδυασμού (από τα πρώτα runs που θα κάνετε). Το “κοντά” σε αυτό το στάδιο μπορούμε να το εκτιμήσουμε μόνο εμπειρικά, επαναλαμβάνοντας με διάφορα `delta` έτσι ώστε στην τελική επιλογή `delta` οι καλύτεροι συνδυασμοί να έχουν ποσοστό επιτυχιών κοντά στη μονάδα και οι χειρότεροι καμία. 
# 
# ----
# 
# Επειδή η όλη διαδικασία της εύρεσης του βέλτιστου συνδυασμού απαιτεί ένα είδος εξαντλητικής αναζήτησης, αφού συνολικά θα χρειαστεί να τρέξουμε $96 * maxrounds$ γενετικούς αλγόριθμους, δημιουργήσαμε μια κύρια wrapper συνάρτηση η οποία δέχεται όλες τις απαραίτητες παραμέτρους για την εκτέλεση τόσο των στρατηγικών εξέλιξης όσο και για κάποιες επιπλέον συναρτήσεις που θα αναφέρουμε στην συνέχεια. 

# In[480]:


def evolution_with_stats(ge_with_stats, npop, toolbox, mu, lambda_, ngen, cxpb, mutpb, rounds, goal, delta, verbose=False):
    MAX_ROUNDS = rounds

    # Absolute Criteria
    avg_min = []
    avg_evals = []
    avg_time = []
    
    # Relevant Criteria
    successes = []
    s_avg_gens = []
    s_avg_min = []
    s_avg_evals = []
    
    for i in range(MAX_ROUNDS):
        if not gstrategy_name(ge_with_stats) == "eaSimple":
            args = {"mu" : mu, "lambda_" : lambda_}
        else:
            args = {}
        
        # Start Stopwatch
        stopwatch = alarm()
        pop, log = ge_with_stats(npop=npop, toolbox=toolbox, ngen=ngen, cxpb=cxpb, mutpb=mutpb, **args)
        avg_time.append(alarm(stopwatch))
        # Stop Stopwatch
        
        # Absolute Criteria
        min_fit, nevals = abs_crit(pop, log)
        avg_min.append(min_fit)
        avg_evals.append(nevals)  
        
        #Relevant Criteria
        success, s_gens, s_min, s_evals = rel_crit(log, goal, delta)
        if success:
            successes.append(success)
            s_avg_gens.append(s_gens)
            s_avg_min.append(s_min)
            s_avg_evals.append(s_evals)
        
        if verbose:
            flag = "Sucess" if success else "Failure"
            print("[Round #{0}] {1}".format(i, flag), success, s_gens, s_min, s_evals)
    
    # Absolute Criteria
    avg_time = np.mean(avg_time)
    avg_evals = np.mean(avg_evals)
    avg_min = np.mean(avg_min)
    
    # Relevant Criteria
    if successes:
        successes = np.sum(successes)
        s_avg_gens = np.mean(s_avg_gens)
        s_avg_min = np.mean(s_avg_min)
        s_avg_evals = np.mean(s_avg_evals)
    else:
        successes = 0
        s_avg_gens = None
        s_avg_min = None
        s_avg_evals = None
        
    return successes, s_avg_min, s_avg_evals, s_avg_gens, avg_evals, avg_min, avg_time


# Για να αποφύγουμε να μεγαλώσουμε αρκετά το μέγεθος της συνάρτησης, φτιάξαμε τις εξής βοηθητικές συναρτήσεις:
# - alarm(): Μετράει τον χρόνο που χρειάζεται για να τρέξει ο γενετικός αλγόριθμος
# - abs_crit: Επιστρέφει τις τιμές των απόλυτων κριτηρίων για κάθε γύρο
# - rel_crit: Επιστρέφει τις τιμές των σχετικών κριτηρίων για κάθε γύρο
# 
# Αφού λοιπόν ολοκληρωθούν όλοι οι γύροι υπολογίζουμε και επιστρέφουμε τα τελικά μεγέθη.
# 
# ```python
# avg_min, avg_evals, avg_time, successes, success_avg_gen, success_avg_min, success_avg_evals
# ```

# In[13]:


def alarm(prev_time=None):
    if not prev_time:
        return time.time()
    else:
        return (time.time()-prev_time)


# In[14]:


def abs_crit(pop, log):
    min_fit = get_min_fit(pop)
    nevals = log.select("nevals")
    
    return min_fit, np.sum(nevals)

def get_min_fit(pop):
    (best_ind, ) = tools.selBest(pop, k=1)
    (min_fit, ) = best_ind.fitness.values
    
    return min_fit


# In[15]:


def rel_crit(logbook, goal, delta):
    success = None
    success_gen = None
    success_min = None
    success_evals = None
    
    evals_cnt = 0
    for gen, log in enumerate(logbook):
        evals_cnt += log["nevals"]
        
        if log["min"] <= (goal + delta):
            success = 1
            success_gen = gen
            success_min = log["min"]
            success_evals = evals_cnt
            
            break
    return (success, success_gen, success_min, success_evals)


# Το πρώτο μας βήμα θα είναι να κατασκευάσουμε τα `containers` που θα περιγράφουν τις οντότητες/άτομα και τον πληθυσμό. Το `toolbox` θα  τροποποιείται κάθε φορά ώστε να αντιστοιχεί στον συνδυασμό των τελεστών και της στρατηγικής εξέλιξης που θέλουμε να εξετάσουμε κάθε φορά. Η μόνη εγγραφή που θα μείνει ανεπηρέαστη σε όλους τους συνδυασμούς είναι η `evaluate`, που απλά υπολογίζει την τιμή της συνάρτησης που θέλουμε να ελαχιστοποιήσουμε. 
# 
# Επιλέξαμε να μην αναλύσουμε περαιτέρω τις βασικές εντολές κατασκευής των δομών καθώς πιστεύουμε ότι θεωρούνται ήδη γνωστές απο το εργαστήριο.

# In[16]:


NUM_VARS = 4

creator.create( "min_fitness", base.Fitness , weights=(-1.0,))
creator.create( "individual_container", list , fitness= creator.min_fitness)
toolbox = base.Toolbox()
toolbox.register("init_value", np.random.uniform, -100, 100)
toolbox.register("individual", tools.initRepeat, creator.individual_container, toolbox.init_value, NUM_VARS)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def eval_corana(ind):
    sum = corana(ind)
    
    return (sum,)

toolbox.register("evaluate", eval_corana)


# Στην συνέχεια θα ορίσουμε τα ορίσματα που θα χρειαστεί να δόσουμε στους τελεστές μετάλλαξης ώστε στην συνέχεια να τα εγγράψουμε στο `toolbox` του γενετικού αλγόριθμου. Ευτυχώς η Python μας παρέχει έναν πολύ εύκολο τρόπο να περνάμε τα ορίσματα μέσω ενός λεξικού, χωρίς να χρειάζεται να γράφουμε ένα-ένα τα ονόματα των μεταβλητών στην συνάρτηση που θέλουμε να εκτελέσουμε. Τα ορίσματα αυτά λέγονται `kargs` (keyword-arguments)
# 
# ------
# Στα τρία παρακάτω κελιά φαίνονται όλοι οι συνδυασμοί τιμών που θέλουμε να παράξουμε με τις υπεπαραμέτρους των γενετικών τελεστών. Έχουμε διαμορφώσει τα ορίσματα σε μορφή λεξικού και έχουμε γράψει βοηθητικές συναρτήσεις για την παραγωγή των `k[eyword]arg[ument]s`.

# ##### Crossover/Mate Combinations

# In[17]:


kargs_cxBlend = {
    "function" : tools.cxBlend,
    "args" : {
        "alpha" : [1,5]
    }
}

kargs_cxSimulatedBinary = {
    "function" : tools.cxSimulatedBinary,
    "args" : {
        "eta" : [0.2, 0.8]
    }
}

mate_ops = [kargs_cxBlend, kargs_cxSimulatedBinary]

def args_product(d):
    keys = d.keys()
    for element in itertools.product(*d.values()):
        yield dict(zip(keys, element))

for mate in mate_ops:
    function = mate["function"]
    for mate_flag, args in enumerate(args_product(mate["args"])):
        toolbox.register("mate", function=function, **args)
        print(mate_flag, toolbox.mate)


# ##### Mutation Combinations

# In[18]:


kargs_mutGaussian = {
    "function" : tools.mutGaussian,
    "args" : {
        "mu" : [0,20],
        "sigma" : [1,10],
        "indpb" : [0.5, 0.5]
    }
}

kargs_mutPolynomialBounded = {
    "function" : mutPolynomialBounded,
    "args" : {
        "eta" : [0.2, 0.2],
        "low" : [-5, -100],
        "up" : [5, 100],
        "indpb" : [0.5, 0.5]
    } 
}

mutation_ops = [kargs_mutGaussian, kargs_mutPolynomialBounded]

def args_perm(d):
    keys = d.keys()
    fillvalue = d["indpb"][0] if "indpb" in d else None
    for element in itertools.zip_longest(*d.values(), fillvalue=fillvalue):
        yield dict(zip(keys, element))

for mutation in mutation_ops:
    function = mutation["function"]
    for mutation_flag, args in enumerate(args_perm(mutation["args"])):
        toolbox.register("mutation", function=function, **args)
        print(mutation_flag, toolbox.mutation)


# ##### Selection Combinations

# In[19]:


kargs_selTournament = {
    "function" : tools.selTournament,
    "args" : {
        "tournsize" : [5, 30]
    }
}

selection_ops = [kargs_selTournament]

for selection in selection_ops:
    function = selection["function"]
    
    for selection_flag, args in enumerate(args_product(selection["args"])):
        toolbox.register("select", function=function, **args)
        print(selection_flag, toolbox.select)


# #####  Evolution Strategy Wrappers

# Αφού λοιπόν καταφέραμε να παράξουμε τους συνδυασμούς των γενετικών τελεστών θα γράψουμε τους επόμενους 3 wrapper για τις στρατηγικές εξέλιξης που μας ζητείται να δοκιμάσουμε. Ο πληθυσμός καθώς και οι πιθανότητες μετάλλαξης και διαστάυρωσης είναι ίδιες και για τους 3 γενετικούς αλγόριθμους. Στην περίπτωση των `eaMuPlusLambda, eaMuCommaLambda` έχουμε δύο ακόμα υπεπαραμέτρους τα `mu,lambda_`, οι οποίες ειναι ίδιες και για τους δύο αλγόριθμους.

# In[477]:


def eaSimple_with_stats(npop, toolbox, cxpb, mutpb, ngen):
    pop = toolbox.population(n=npop)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min)
    
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=stats, verbose=False)

    return pop, log


# In[478]:


def eaMuPlusLambda_with_stats(npop, toolbox, mu, lambda_, cxpb, mutpb, ngen):
    pop = toolbox.population(n=npop)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min)
    
    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu=mu , lambda_=lambda_ , cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=stats, verbose=False)

    return pop, log


# In[479]:


def eaMuCommaLambda_with_stats(npop, toolbox, mu, lambda_, cxpb, mutpb, ngen):
    pop = toolbox.population(n=npop)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min)
    
    pop, log = algorithms.eaMuCommaLambda(pop, toolbox, mu=mu , lambda_=lambda_ , cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=stats, verbose=False)

    return pop, log


# Περνάμε πλέον στο τελικό στάδιο όπου όλα τα επιμέρους κομμάτια πρέπει να ενωθούν για να δημιουργήσουν την τελική συνάρτηση εύρεσης του βέλτιστου συνδυασμού. Αν παρατηρήσατε προηγουμένως έχουμε σκοπίμα γράψει τους βρόγχους για την παραγωγή των συνδυασμών των τελεστών, μάζι την αρίθμηση τους (0,1). Το `0` θα αντιστοιχεί στην "Low" επιλογή, ενω το `1` στην "High". Επίσης γράψαμε δύο βοηθητικές συνάρτησεις που θα μας δίνουν το όνομα της συνάρτησης του γενετικού τελεστή και της στρατηγικής.

# In[207]:


test_operator = toolbox.mate

def goperator_name(op, flag=None):
    if not (flag == None):
        flag = "High" if (flag == 1) else "Low"
        
        return str(op).split("(")[1].split()[1] + flag
    else:
        return str(op).split("(")[1].split()[1]

print(goperator_name(test_operator,0))
print(goperator_name(test_operator,1))


# In[24]:


test_strategy = '<function eaSimple_with_stats at 0x1187ddea0>'

def gstrategy_name(strategy):
    return str(strategy).split()[1].split("_")[0]

gstrategy_name(test_strategy)


# In[43]:


def listify(res):
    ret = []
    for val in res:
        if isinstance(val, (str)):
            ret.append(val)
        elif isinstance(val, (float)):
            ret.append(val)
        elif isinstance(val, (int)):
            ret.append(val)
        else:
            ret.append(val)
    return ret


# #### Συνάρτηση Εύρεσης Βέλτιστου Συνδυασμού (Βλ. Σημείωση πριν το Ερώτημα 2)

# In[511]:


ge_args = {
    "ge_with_stats" : None,
    "npop" : 100,
    "toolbox" : toolbox,
    "mu" : 50,
    "lambda_" : 100,
    "ngen" : 100,
    "cxpb" : 0.5,
    "mutpb" : 0.2,
    "rounds" : 10,
    "goal" : 0.0,
    "delta" : 0.0005,
    "verbose" : False
}

strategy_ops = [eaSimple_with_stats, eaMuCommaLambda_with_stats, eaMuPlusLambda_with_stats]
mate_ops = [kargs_cxBlend, kargs_cxSimulatedBinary]
mutation_ops = [kargs_mutGaussian, kargs_mutPolynomialBounded]
selection_ops = [kargs_selTournament]

def gridsearch(ge_args, strategy_ops, mate_ops, mutation_ops, selection_ops, verbose=False):
    metrics = []
    
    ##### Strategy Loop
    for strategy in strategy_ops:
        ge_args["ge_with_stats"] = strategy

        ##### Crossover/Mate Loop
        for mate in mate_ops:
            mate_function = mate["function"]
            for mate_flag, mate_args in enumerate(args_product(mate["args"])):
                toolbox.register("mate", function=mate_function, **mate_args)
                ##### End of Crossover/Mate Loop

                ##### Mutation Loop
                for mutation in mutation_ops:
                    mutation_function = mutation["function"]
                    for mutation_flag, mutation_args in enumerate(args_perm(mutation["args"])):
                        toolbox.register("mutate", function=mutation_function, **mutation_args)
                        ##### End of Mutation Loop

                        ##### Selection Loop
                        for selection in selection_ops:
                            selection_function = selection["function"]
                            for selection_flag, selection_args in enumerate(args_product(selection["args"])):
                                toolbox.register("select", function=selection_function, **selection_args)
                                ##### End of Selection Loop

                                # Get Operator Names
                                mate_name = goperator_name(toolbox.mate, mate_flag)
                                mutation_name = goperator_name(toolbox.mutate, mutation_flag)
                                selection_name = goperator_name(toolbox.select, selection_flag)
                                strategy_name = gstrategy_name(strategy)

                                # Metrics
                                lmetrics = listify(evolution_with_stats(**ge_args))

                                # Operator Name
                                operator_name = ",".join([mate_name, mutation_name, selection_name])

                                # Strategy Args
                                strategy_args = [strategy_name, str(ge_args["npop"])]
                                if not (strategy_name == "eaSimple"):
                                    strategy_args.append(str(ge_args["mu"]))
                                    strategy_args.append(str(ge_args["lambda_"]))

                                strategy_name = " ".join(strategy_args)

                                ret = [operator_name, strategy_name, *lmetrics]
                                
                                if verbose:
                                    print(ret)
                                
                                metrics.append(ret)
    return metrics


# In[39]:


metrics = gridsearch(ge_args, strategy_ops, mate_ops, mutation_ops, selection_ops, verbose=True)


# In[40]:


columns = ['operators', 'strategy', 'successes', 's.avg.min', 's.avg.evals', 's.avg.gens', 'avg.evals', 'avg.min', 'avg.time']
frame1 = report(metrics, columns_names=columns)
frame1.sort_values(by=['successes'], ascending=False)


# #### Τελική αναζήτηση και σχολιασμός
# 
# Ο παραπάνω πίνακας παρουσιάζει τα αποτελέσματα που προέκυψαν από το `gridsearch` εξετάζοντας όλες τους συνδυασμούς των παραμέτρων και όλες τις στρατηγικές. Επιλέξαμε να ταξινομήσουμε τον πίνακα με βάση τον αριθμό των επιτυχημένων προσπαθειών αλλά και της μέσης τιμής των ελαχίστων. Δυστυχώς μόνο δύο συνδυασμοί κατάφεραν να πετύχουν το απόλυτο $10/10$ ενώ υπήρχαν άλλες 3  προσπάθειες με επιτυχία $9/10$. Κατά κύριο λόγο ο `eaMuCommaLambda` μαζί με τον `eaSimple` ήταν εκείνοι που βρέθηκαν περισσότερες φορές στην αρχική δεκάδα, πράγμα που μας κατευθύνει ενδεχομένως για την αναζήτηση του βέλτιστου συνδυασμού. Παρόλα αυτά αξίζει να σημειώσουμε ότι ο απλός γενετικός αλγοριθμός, πράγμα προφανές βέβαια, χρειάζεται μέχρι και σε $25\%$ λιγότερα evaluations. 

# #### Παρατηρήσεις #1

# Επειδή η γενική μορφή των δύο αλγορίθμων, `eaMuCommaLambda, eaMuPlusLambda`, δεν διαφέρει ιδιαίτερα θα έπρεπε να μας παραξενεύει το γεγονός ο πρώτος να βρίσκεται πολύ συχνά στην λίστα των δέκα καλύτερος, ενώ ο δεύτερος το πολύ μια. Για να εξετάσουμε περαιτέρω αυτό το ζήτημα θα εκτελέσουμε ένα νέο gridsearch με μικρότερο αριθμό γύρων, αυξάνοντας όμως τις δύο κοινές υπεπαραμέτρους τους `mu,lambda_` .

# In[482]:


ge_args = {
    "ge_with_stats" : None,
    "npop" : 100,
    "toolbox" : toolbox,
    "mu" : 50,
    "lambda_" : 100,
    "ngen" : 100,
    "cxpb" : 0.5,
    "mutpb" : 0.2,
    "rounds" : 10,
    "goal" : 0.0,
    "delta" : 0.0005,
    "verbose" : False
}

strategy_ops = [eaMuPlusLambda_with_stats, eaMuCommaLambda_with_stats]
mate_ops = [kargs_cxBlend, kargs_cxSimulatedBinary]
mutation_ops = [kargs_mutGaussian, kargs_mutPolynomialBounded]
selection_ops = [kargs_selTournament]


# In[46]:


metrics2 = gridsearch(ge_args, strategy_ops, mate_ops, mutation_ops, selection_ops)


# In[44]:


frame2 = report(metrics2, columns_names=columns)
frame2.sort_values(by=['successes', 's.avg.gens'], ascending=False)


# #### Παρατηρήσεις #2

# Παρατηρούμε ότι η στρατηγική εξέλιξης που κυριαρχεί στις καλύτερες προσπάθεις είναι eaMuCommaLambda_with_stats. Παρόλα αυτά θα συνεχίσουμε την αναζήτηση μας μεταξύ των δύο αφού πρώτα αποκλίσουμε κάποιους γενετικούς τελεστές που δεν συμβάλλουν στην καλή επίδοση των αποτελεσμάτων. Οι παρατηρήσεις που θα παραθέσουμε προκύπτουν σε συνδυασμό με τα αποτελέσματα που έχουμε απο τις δύο εκτελέσης του `gridsearch`. 
# 
# - Ο συνδυασμός High τελεστών διασταύρωσης και μετάλλαξης συνήθως οδηγούν σε ολική αποτυχία (0/0)
# - Ο συνδυασμός Low και των τριών τελεστών οδηγεί σε πλήρη λύση (10/10) και μάλιστα με μικρότερο αριθμό γενεών.
# - Ο mutGaussianHigh δεν οδηγεί σχεδόν ποτέ σε πάνω από 6/10.
# - Ο cxSimulatedBinary[High/Low] φαίνεται να κυριαρχεί στις καλύτερες λύσεις
# 
# Σύμφωνα με τις παρατηρήσεις θα ξεκινήσουμε ένα νέο `gridsearch` όπου θα ισχύουν τα εξής:
# 
# - Οι γύροι θα παραμείνουν στους 10
# - Οι υπερπαράμετροι mu, lambda_, ngens θα παραμείνουν ως έχουν
# - Crossover: cxBlendLow, cxSimulatedBinary[Low/High]
# - Mutation: mutGaussianLow, mutPolynomialBounded[Low/High]
# - Ο τελεστής επιλογής θα παραμείνει το ίδιο

# In[47]:


kargs_cxBlendLow = {
    "function" : tools.cxBlend,
    "args" : {
        "alpha" : [1]
    }
}

kargs_mutGaussianLow = {
    "function" : tools.mutGaussian,
    "args" : {
        "mu" : [0],
        "sigma" : [1],
        "indpb" : [0.5]
    }
}


# In[483]:


ge_args = {
    "ge_with_stats" : None,
    "npop" : 100,
    "toolbox" : toolbox,
    "mu" : 75,
    "lambda_" : 150,
    "ngen" : 100,
    "cxpb" : 0.5,
    "mutpb" : 0.2,
    "rounds" : 10,
    "goal" : 0.0,
    "delta" : 0.0005,
    "verbose" : False
}

strategy_ops = [eaMuPlusLambda_with_stats, eaMuCommaLambda_with_stats]
mate_ops = [kargs_cxBlendLow, kargs_cxSimulatedBinary]
mutation_ops = [kargs_mutGaussianLow, kargs_mutPolynomialBounded]
selection_ops = [kargs_selTournament]


# In[51]:


metrics3 = gridsearch(ge_args, strategy_ops, mate_ops, mutation_ops, selection_ops)


# In[642]:


frame3


# In[52]:


frame3 = report(metrics3, columns_names=columns)
frame3.sort_values(by=['successes', 's.avg.evals'], ascending=False)


# ##### Παρατηρήσεις #3

# Είναι βέβαιο πλέον ότι ο `eaMuCommaLambda` είναι εκείνος που αποδίδει καλύτερα. Εκτός όμως απο την επιλογή που θα κάνουμε εκ των δύο, μπορούμε να εξάγουμε και άλλα σημαντικά στοιχεία για την μετέπειτα αναζήτηση του βέλτιστου συνδυασμού. 
# 
# - selTournamentLow αποδίδει σε γενικές γραμμές καλύτερα
# - Ο συνδυασμός όλα High οδηγεί σε αποτυχία
# - Ο αριθμός των evalutions είναι σχεδόν ίδιος σε όλες τις περιπτώσεις
# - Οι συνδυασμοί [cxBlendLow, Low, Low, eaMuPlusLambda], [Low, Low, Low, eaMuCommaLambda ] καταλήγουν πιο γρήγορα σε λύση.
# 
# Θα επιλέξουμε λοιπόν τους αλγόριθμους eaMuCommaLambda,eaSimple με όλους τους Low γενετικούς τελεστές ως την τελική μας δοκιμή για την ανάδειξη της βέλτιστης μεθόδου. Επίσης θα μειώσουμε ακόμα περισσότερο το `delta` αυτή την φορά ενώ θα αυξήσουμε τους γύρους στο 20.

# In[54]:


kargs_cxBlend = {
    "function" : tools.cxBlend,
    "args" : {
        "alpha" : [1]
    }
}

kargs_cxSimulatedBinary = {
    "function" : tools.cxSimulatedBinary,
    "args" : {
        "eta" : [0.2]
    }
}

kargs_mutGaussian = {
    "function" : tools.mutGaussian,
    "args" : {
        "mu" : [0],
        "sigma" : [1],
        "indpb" : [0.5]
    }
}

kargs_mutPolynomialBounded = {
    "function" : mutPolynomialBounded,
    "args" : {
        "eta" : [0.2],
        "low" : [-5],
        "up" : [5],
        "indpb" : [0.5]
    } 
}

kargs_selTournament = {
    "function" : tools.selTournament,
    "args" : {
        "tournsize" : [5]
    }
}


# In[484]:


ge_args = {
    "ge_with_stats" : None,
    "npop" : 100,
    "toolbox" : toolbox,
    "mu" : 75,
    "lambda_" : 150,
    "ngen" : 100,
    "cxpb" : 0.5,
    "mutpb" : 0.2,
    "rounds" : 20,
    "goal" : 0.0,
    "delta" : 0.0000005,
    "verbose" : False
}

strategy_ops = [eaSimple_with_stats, eaMuCommaLambda_with_stats]
mate_ops = [kargs_cxBlend, kargs_cxSimulatedBinary]
mutation_ops = [kargs_mutGaussian, kargs_mutPolynomialBounded]
selection_ops = [kargs_selTournament]


# In[57]:


metrics4 = gridsearch(ge_args, strategy_ops, mate_ops, mutation_ops, selection_ops)


# In[58]:


frame4 = report(metrics4, columns_names=columns)
frame4.sort_values(by=['successes', 's.avg.evals'], ascending=False)


# ##### Παρατηρήσεις 5

# Μιας και ο τελεστής διαστάυρωσης `cxSimulatedBinaryLow` κυριαρχεί στις πρώτες θέσεις θα είναι αυτός που θα επιλέξουμε στον τελικό μας αλγόριθμο. Το ίδιο συμβαίνει και με τον `mutGaussianLow`. Όσο για τον γενετικό αλγόριθμο θα εκτελέσουμε ένα ακόμα `gridsearch` με αριθμό γύρων 30 και ακόμα μικρότερο `delta` ώστε να είμαστε βέβαιοι ότι θα κάνουμε την καλύτερη επιλογή.

# In[485]:


ge_args = {
    "ge_with_stats" : None,
    "npop" : 100,
    "toolbox" : toolbox,
    "mu" : 75,
    "lambda_" : 150,
    "ngen" : 100,
    "cxpb" : 0.5,
    "mutpb" : 0.2,
    "rounds" : 30,
    "goal" : 0.0,
    "delta" : 0.000000005,
    "verbose" : False
}

strategy_ops = [eaSimple_with_stats, eaMuCommaLambda_with_stats]
mate_ops = [kargs_cxSimulatedBinary]
mutation_ops = [kargs_mutGaussian]
selection_ops = [kargs_selTournament]


# In[62]:


metrics5 = gridsearch(ge_args, strategy_ops, mate_ops, mutation_ops, selection_ops)


# In[63]:


frame5 = report(metrics5, columns_names=columns)
frame5.sort_values(by=['successes', 's.avg.evals'], ascending=False)


# Θα ακολουθήσουμε την πλειοψηφία και θα επιλέξουμε τον `eaMuCommaLambda` για αλγόριθμο εξέλιξης. Η πορεία των επιλογών μας είναι ξεκάθαρη μέσα απο την σειρά των Παρατηρήσεων (1-5) που έχουμε παραθέσει παραπάνω. Παρόλα που και ο `eaSimple` έχει ποσοστό επιτυχίας 50%, ο δεύτερος είναι συντρηπτικά καλύτερος.

# 
# ### Τελική βελτιστοποίηση
# 

# In[206]:


# Best GA 
toolbox.register("mate", tools.cxSimulatedBinary, eta=0.2)
toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=1.0, indpb=0.5)
toolbox.register("select", tools.selTournament, tournsize=5)


# #### Βελτιστοποίηση πιθανοτήτων διασταύρωσης και μετάλλαξης
# 
# Έχουμε ήδη επιλέξει τον καταλληλότερο συνδυασμό των γενετικών τελεστών και του γενετικού αλγόριθμου. Το επόμενο βήμα θα είναι να βρούμε τις πιθανότητες των τελεστών διασταύρωσης και μετάλλαξης στο διάστημα $[0.05, 0.9]$ που ορίζονται στον `eaMuCommaLambda`.

# Θα ξεκινήσουμε με έναν μικρό αριθμό γύρων και με `delta` στο $0.000000005$.

# In[486]:


ge_args = {
    "ge_with_stats" : eaMuCommaLambda_with_stats,
    "npop" : 100,
    "toolbox" : toolbox,
    "mu" : 75,
    "lambda_" : 150,
    "ngen" : 100,
    "cxpb" : None,
    "mutpb" : None,
    "rounds" : 10,
    "goal" : 0.0,
    "delta" : 0.000000005,
    "verbose" : False
}

def probs_gridsearch(ge_args, cxpbs, mutpbs, verbose=False):
    metrics = []
    for cxpb in cxpbs:
        for mutpb in mutpbs:
            ge_args["cxpb"] = cxpb
            ge_args["mutpb"] = mutpb
            
            if (cxpb + mutpb > 1.0):
                continue
            
            lmetrics = listify(evolution_with_stats(**ge_args))
            ret = [cxpb, mutpb, *lmetrics]

            if verbose:
                print(ret)
            
            metrics.append(ret)
    return metrics


# In[122]:


cxpbs = np.linspace(0.5, 0.9, 4)
mutpbs = np.linspace(0.1, 0.4, 4)
print(cxpbs, mutpbs, sep="\n")


# In[123]:


prob = probs_gridsearch(ge_args, cxpbs, mutpbs)


# In[124]:


probs_columns = ['cxpb', 'mutpb', 'successes', 's.avg.min', 's.avg.evals', 's.avg.gens', 'avg.evals', 'avg.min', 'avg.time']

prob_frame = report(prob, columns_names=probs_columns)
prob_frame.sort_values(by=['successes', 's.avg.evals'], ascending=False)


# Από τα αποτελέσματα καταλαβαίνουμε οτι οι ιδανικές τιμές για την πιθανότητα `cxpb` βρίσκονται στο διάστημα [0.5,0.6], ενώ η `mutpb` [1.8, 2.2]

# In[99]:


cxpbs = np.linspace(0.5, 0.6, 5)
mutpbs = np.linspace(0.18, 0.22, 5)
print(cxpbs, mutpbs)


# In[ ]:


ge_args = {
    "ge_with_stats" : eaMuCommaLambda_with_stats,
    "npop" : 100,
    "toolbox" : toolbox,
    "mu" : 75,
    "lambda_" : 150,
    "ngen" : 100,
    "cxpb" : None,
    "mutpb" : None,
    "rounds" : 10,
    "goal" : 0.0,
    "delta" : 0.000000005,
    "verbose" : False
}

prob2 = probs_gridsearch(ge_args, cxpbs, mutpbs)


# In[104]:


prob_frame2 = report(prob2, columns_names=probs_columns)
prob_frame2.sort_values(by=['successes', 's.avg.evals'], ascending=False)


# Θα συνεχίσουμε την αναζήτηση μας σε ένα ακόμα **μικρότερο διάστημα** για να βρούμε εν τέλει τον καλύτερο συνδυασμό των πιθανοτήτων διασταύρωσης και μετάλλαξης.

# In[132]:


cxpbs = np.linspace(0.48, 0.56, 5)
mutpbs = np.linspace(0.19, 0.205, 5)
print(cxpbs, mutpbs)


# In[ ]:


ge_args = {
    "ge_with_stats" : eaMuCommaLambda_with_stats,
    "npop" : 100,
    "toolbox" : toolbox,
    "mu" : 75,
    "lambda_" : 150,
    "ngen" : 100,
    "cxpb" : None,
    "mutpb" : None,
    "rounds" : 10,
    "goal" : 0.0,
    "delta" : 0.000000005,
    "verbose" : False
}

prob3 = probs_gridsearch(ge_args, cxpbs, mutpbs)


# In[134]:


prob_frame3 = report(prob3, columns_names=probs_columns)
prob_frame3.sort_values(by=['successes', 's.avg.evals'], ascending=False)


# 
# #### Εύρεση βέλτιστης (ελάχιστης) τιμής της συνάρτησης με τον ΓΑ
# 
# Θα εκτελέσουμε ένα τελευταίο run του βέλτιστου αλγόριθμου που προέκυψε στα προηγούμενα βήματα με ένα μεγάλο αριθμό γενεών και πληθυσμού για να πάρουμε μια βέλτιστη τιμή για τη συνάρτηση (σε λογικά χρονικά πλαίσια). Ως συνδυασμό πιθανοτήτων θα κρατήσουμε αυτόν που πέτυχε την καλύτερη `avg.min` -> 0.54	0.19375.
# 

# In[150]:


if __name__ == "__main__":
    pop = toolbox.population(n=500)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min)
    
    stopwatch = alarm()
    pop, log = algorithms.eaMuCommaLambda(pop, toolbox, mu=100, lambda_=250, ngen=500, cxpb=0.54, mutpb=0.19375, stats=stats, verbose=False)
    stopwatch = alarm(stopwatch)
    
    min_fit, nevals = abs_crit(pop, log)
    
    print("Best (Fitness) Value: {0}, Evals: {1}, Time: {2}".format(min_fit, nevals, stopwatch))
    


# In[153]:


import numpy
import numpy as np
from math import sqrt, pi, exp, cos, e
import random


# ### Σημείωση
# Στα κελιά όπου οι υπεπαράμετροι `mu,lambda_` των γενετικών αλγορίθμων έχουν αλλάξει, οι πίνακες των αποτελεσμάτων συνεχίζουν να δείχνουν σταθερά 50,100. Όταν είχαμε πρώτο-κατασκεύασει την `gridsearch` δεν είχαμε προσθέσει την δυνατότητα τροποποίησης των δύο, και κατά την σύνταξη των ονομάτων που θα εμφανιζόντουσαν στο DataFrame προσθέταμε δίπλα απο το όνομα σταθερά αυτές τις τιμές. Οπότε μετάπειτα που προσθέσαμε αυτήν την δυνατότητα ξεχάσαμε να αλλάξουμε τον τρόπο που γράφεται το strategy.
# 
# **Πριν**
# ```python
#     # Strategy Args
#     strategy_args = [strategy_name, "200"]
#     if not (strategy_name == "eaSimple"):
#         strategy_args.append("50")
#         strategy_args.append("100")
# ```
# 
# **Μετά**
# ```python
#     # Strategy Args
#     strategy_args = [strategy_name, "200"]
#     if not (strategy_name == "eaSimple"):
#         strategy_args.append(str(ge_args["mu"]))
#         strategy_args.append(str(ge_args["lambda_"]))
# ```
# 
# Το αναφέρουμε απλά για να μην δημιουργηθεί παρεξήγηση ως προς την λειτουργικότητα του αλγορίθμου.

# 
# ## Μέρος 2. Μελέτη κλιμακούμενης συνάρτησης
# 
# Όπως είδαμε και στο παράδειγμα του 0-1 Knapsack, πολλά προβλήματα μπορεί να λύνονται εύκολα σε μικρές διαστάσεις αλλά γίνονται δυσκολότερα όσο οι διαστάσεις μεγαλώνουν. Συνεπώς, μια επιθυμητή ιδιότητα για έναν αλγόριθμο είναι να μπορεί να αντιμετωπίζει την κλιμάκωση των προβλημάτων. Η κλιμακούμενη συνάρτηση που μας ζητείται να ελαχιστοποιήσουμε είναι `Ackley1`. Παρακάτω έχουμε παραθέσει δύο συναρτήσεις με διαφορετικό σύνολο εντολών. Η πρώτη χρησιμοποιεί μαθηματικές συναρτήσεις απο την βιβλιοθήκη `math`, ενώ η δεύτερη είναι εξολοκλήρου γραμμένη με την χρήση της `numpy` βιβλιοθήκης. Ο λόγος που έχουμε και τις δύο εκδοχές είναι γιατί η πρώτη θα χρησιμοποιηθεί καθ όλη την διάρκεια του δεύτερου ερωτήματος, ενώ η δεύτερη θα μας βοηθήσει στην **3D** αναπαράσταση. 

# In[503]:


def ackley1(x):
    dim = len(x)
    
    exp1 = (-0.02) * sqrt((1/dim) * sum([i**2 for i in x]))
    exp1 = exp(exp1)

    exp2 = (1/dim) * sum([cos(2 * pi * i) for i in x])
    exp2 = exp(exp2)

    return (-20) * exp1 - exp2 + 20 + e


# In[611]:


def ackley1_np(x, dim=2):
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    
    try:
        res = (-20 * np.exp(-0.02 * np.sqrt((1/dim) * np.sum(x**2, 1))) - np.exp((1/dim) * np.sum(np.cos(2 * np.pi * x), 1)) + 20 + np.e) 
    except:
        res = (-20 * np.exp(-0.02 * np.sqrt((1/dim) * np.sum(x**2))) - np.exp((1/dim) * np.sum(np.cos(2 * np.pi * x))) + 20 + np.e) 
    
    return res


# Θα τρέξουμε μια απλή επανάληψη για να σας αποδείξουμε ότι τα αποτελέσματα και των δύο εκδοχών είναι παρόμοια. Απαραίτητη προυπόθεση για να συμβαίνει αυτό είναι ο `numpy` πίνακας να είναι τύπου `float64` έστω να μην έχουμε θέματα ακρίβειας, κάτι που φροντίζει αυτόματα η `math`.

# In[627]:


def validation(func1, func2, iterations):
    succ = 0

    for i in range(iterations):
        rand_2d_vec = [np.random.uniform(-35,35), np.random.uniform(-35,35)]

        if func1(rand_2d_vec) == func2(rand_2d_vec):
            succ += 1
    return ("Success: {}/{}".format(succ, iterations))

validation(ackley1, ackley1_np, 5000)


# 
# ### Για D=2
# 
# - α) Εκτυπώστε ένα “3D” γράφημα της συνάρτησης $f(x1,x2)$ και περιγράψτε σύντομα τη μορφή της (βέλτιστα, κοίλα, κοιλάδες, λεκάνες, ασυνέχειες κλπ)

# In[183]:


from mpl_toolkits import mplot3d

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import itertools

def grid3d(f, low, upper, num=100):
    x = np.linspace(-35, 35, 100)
    y = np.linspace(-35, 35, 100)
    z = np.array(list(itertools.product(x, y)))   
    
    X, Y = np.meshgrid(x,y)
    
    return(X,Y,f(z, dim=2).reshape(-1, num))

x, y, z = grid3d(ackley1_np, -35, 35)


# In[193]:


def multiplot3d(data, n=4):
    (x,y,z) = data
    fig = plt.figure(figsize=(20 - n, 20 - n), tight_layout=True)
    
    init_code = int("{0}{0}1".format(n))

    for i in range(n):
        subplot = fig.add_subplot(init_code + i, projection='3d')

        subplot.set_xlabel('x1')
        subplot.set_ylabel('x2')
        subplot.set_zlabel('ackley1')

        subplot.plot_surface(x, y, z, cmap='viridis', rstride=1, cstride=1, linewidth=0)
        
        if (n == 1):
            subplot.view_init(10,20)
        else:
            subplot.view_init(10*(i + random.randint(1,10)), 10*(i + random.randint(1,10)))

    plt.show()


# In[194]:


multiplot3d((x,y,z), 1)


# In[195]:


multiplot3d((x,y,z), 4)


# - β) Με την διαδικασία που ακολουθήσαμε προηγουμένως για τη μη κλιμακούμενη συνάρτηση, βρείτε τον βέλτιστο γενετικό αλγόριθμο και τη βέλτιστη τιμή για το πρόβλημα. Θα αρχίσουμε αρχικοποιώντας τα `kargs` για τους γενετικούς τελεστές, και ορίζοντας τα βασικά `containers` του `deap`.

# In[517]:


kargs_cxBlend = {
    "function" : tools.cxBlend,
    "args" : {
        "alpha" : [1,5]
    }
}

kargs_cxSimulatedBinary = {
    "function" : tools.cxSimulatedBinary,
    "args" : {
        "eta" : [0.2, 0.8]
    }
}

kargs_mutGaussian = {
    "function" : tools.mutGaussian,
    "args" : {
        "mu" : [0,20],
        "sigma" : [1,10],
        "indpb" : [0.5, 0.5]
    }
}

kargs_mutPolynomialBounded = {
    "function" : mutPolynomialBounded,
    "args" : {
        "eta" : [0.2, 0.2],
        "low" : [-5, -100],
        "up" : [5, 100],
        "indpb" : [0.5, 0.5]
    } 
}

kargs_selTournament = {
    "function" : tools.selTournament,
    "args" : {
        "tournsize" : [5, 30]
    }
}

strategy_ops = [eaMuCommaLambda_with_stats, eaMuPlusLambda_with_stats, eaSimple_with_stats]
mate_ops = [kargs_cxBlend, kargs_cxSimulatedBinary]
mutation_ops = [kargs_mutGaussian, kargs_mutPolynomialBounded]
selection_ops = [kargs_selTournament]


# In[518]:


NUM_VARS = 2

creator.create( "min_fitness", base.Fitness , weights=(-1.0,))
creator.create( "individual_container", list , fitness= creator.min_fitness)
toolbox = base.Toolbox()
toolbox.register("init_value", np.random.uniform, -35, 35)
toolbox.register("individual", tools.initRepeat, creator.individual_container, toolbox.init_value, NUM_VARS)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def eval_ackley(ind):
    sum = ackley1(ind)
    
    return (sum,)


# Θα ορίσουμε τα `kargs` που δέχεται ο `gridsearch` αλγόριθμος και θα τρέξουμε την πρώτη μας αναζήτηση του βέλτιστου συνδυασμού. 

# In[519]:


ge_args = {
    "ge_with_stats" : None,
    "npop" : 100,
    "toolbox" : toolbox,
    "mu" : 100,
    "lambda_" : 200,
    "ngen" : 100,
    "cxpb" : 0.5,
    "mutpb" : 0.2,
    "rounds" : 5,
    "goal" : 0.0,
    "delta" : 0.005,
    "verbose" : False
}


# Πριν ξεκινήσουμε θα πρέπει πρώτα να εγγράψουμε την συνάρτηση που θα εκτελεί το `evaluation` του fitness των γονίδιων. Το `fitness` value πλεόν των ατόμων δεν θα υπολογίζεται απλά με την τιμή που επιστρέφει η `Ackley1`, αλλά με επιπλέον περιορισμούς. Ο βασικός τρόπος για να επιβάλουμε περιορισμούς είναι να επιβάλουμε μια ποινή στην τιμή της καταλληλότητας στα άτομα που είναι εκτός των ορίων που έχουμε θέσει. 
# 
# ----
# 
# Αρχικά ορίζουμε δύο συναρτήσεις, τη "feasible" που μας επιστρέφει True αν όλα τα $x_i$ είναι εντός του διαστήματος και False αλλιώς και την "distance" που μας ποσοτικοποιεί πόσο εκτός ορίων είναι ένα άτομο. Συγκεκριμένα επιλέγουμε η απόσταση να είναι το απόλυτο άθροισμα σε όλες τις διαστάσεις της απόστασης από το όριο. Θα μπορούσαμε να κάνουμε και άλλες επιλογές όπως πχ να χρησιμοποιήσουμε μια τετραγωνική συνάρτηση της απόστασης.

# In[520]:


MIN_BOUND = -35.0
MAX_BOUND = 35.0

def feasible(indiv):
    for i in range (len(indiv)) :
        if (indiv [i] < MIN_BOUND) or (indiv [i] > MAX_BOUND):
            return False
    return True

def distance(indiv) :
    dist = 0.0
    for i in range (len(indiv)) :
        penalty = 0
        if ( indiv [i] < MIN_BOUND) : penalty = MIN_BOUND - indiv [i]
        if ( indiv [i] > MAX_BOUND) : penalty = indiv [i] - MAX_BOUND
        dist = dist + penalty
    return dist


# Μια πολύ χρήσιμη μέθοδος που διαθέτει η Python και η DEAP είναι η διακόσμηση συναρτήσεων μέσω διακοσμητών (decorators). Πρόκειται για τη δυνατότητα να τροποποιούμε τη συμπεριφορά μιας συνάρτησης χωρίς να μεταβάλουμε τον κώδικά της αλλά επιτυγχάνοντάς το μέσω μιας άλλης συνάρτησης (του decorator). Με την παρακάτω εντολή μπορούμε τροποποιήσουμε τη συνάρτηση καταλληλότητας `eval_ackley` με την builtin `DeltaPenality`. 
# 
# ---
# 
# Το ζήτημα που ανακύπτει είναι πως θα βρούμε την κατάλληλη τιμή για την σταθερά Δ. Θα μπορούσαμε να πούμε ότι πρόκειται για την τελευταία `valid fitness` τιμή των ατόμων. Κοιτώντας την γραφική παράσταση του πρώτου ερωτήματος παρατηρούμε ότι η τιμή αυτή ειναι 12. Αυτό δεν σημαίνει ότι η σταθέρα θα είναι πάντα ίδια για όλες τις διαστάσεις αλλά θα αυξάνεται γραμμικά καθώς αυξάνονται και οι διαστάσεις. Αν λοιπον σε 2 διαστάστεις έχουμε τιμή 12, τότε σε n θα είναι $6* n$.

# In[521]:


toolbox.register("evaluate", eval_ackley)
toolbox.decorate("evaluate", tools.DeltaPenality (feasible, 12, distance))


# Η DeltaPenality ή ποινή-Δ απαιτεί τουλάχιστον δύο ορίσματα. Το πρώτο πρέπει να επιστρέφει αν ένα άτομο είναι έγκυρο ή όχι, σύμφωνα με τα όρια που έχουμε θέσει. Εμείς θα χρησιμοποιήσουμε τη "feasible" που ορίσαμε γι' αυτό το λόγο. Το δεύτερο όρισμα είναι η σταθερά Δ, δηλαδή η σταθερή ποινή που θα προστεθεί (σε πρόβλημα ελαχιστοποίησης) ή αφαιρεθεί (σε πρόβλημα μεγιστοποίησης) στην τιμή καταλληλότητας ενός ατόμου που είναι εκτός των ορίων που θέλουμε. Ο τρίτος όρος είναι μια επιπλέον ποινή που μπορεί να εφαρμοστεί και που συνήθως την ορίζουμε να είναι ανάλογη του κατά πόσο είναι εκτός ορίων ένα άτομο. Συνολικά δηλαδή θα έχουμε: 
# $$f_i^\mathrm{penalty}(\mathbf{x}) = \Delta - w_i d_i(\mathbf{x})$$

# In[510]:


ametrics = gridsearch(ge_args, strategy_ops, mate_ops, mutation_ops, selection_ops, verbose=True)


# In[515]:


columns = ['operators', 'strategy', 'successes', 's.avg.min', 's.avg.evals', 's.avg.gens', 'avg.evals', 'avg.min', 'avg.time']
aframe1 = report(ametrics, columns_names=columns)
aframe1.sort_values(by=['successes','avg.min'], ascending=False)


# Λόγω του μικρού αριθμού γύρων αλλά και της καλής συμπεριφοράς της συνάρτησης μας, πολλά απο τα αποτελέσματα σημειώνουν πλήρη επιτυχία (5/5). Παρόλα αυτά πολλά απο τα αποτελέσματα είναι ψευδό-σωστά, αν παρατηρήσουμε τις στήλες των σχετικών κριτηρίων. Μπορεί στην στήλη των `successes` να πετυχαίνουμε 5/5, οι περισσότερες όμως απο τις τιμές της `s.avg.min` εμφανίζουν μια σχετική απόκλιση απο το απόλυτο 0.0, και αυτό βέβαια έχει να κάνει τόσο με τον πληθυσμό όσο και απο τον αριθμό των γενεών. Εφόσον ως στόχο έχουμε να εντοπίσουμε τον βέλτιστο συνδυασμό γενετικών τελεστών και αλγόριθμου, το επόμενο μας βήμα θα είναι να μειώσουμε το `delta` και να εκτελέσουμε ένα νέο `gridsearch`.

# In[542]:


ge_args = {
    "ge_with_stats" : None,
    "npop" : 100,
    "toolbox" : toolbox,
    "mu" : 100,
    "lambda_" : 200,
    "ngen" : 100,
    "cxpb" : 0.7,
    "mutpb" : 0.2,
    "rounds" : 5,
    "goal" : 0.0,
    "delta" : 0.00000000005,
    "verbose" : False
}

strategy_ops = [eaSimple_with_stats, eaMuCommaLambda_with_stats, eaMuPlusLambda_with_stats]
mate_ops = [kargs_cxBlend, kargs_cxSimulatedBinary]
mutation_ops = [kargs_mutGaussian, kargs_mutPolynomialBounded]
selection_ops = [kargs_selTournament]


# In[537]:


ametrics2 = gridsearch(ge_args, strategy_ops, mate_ops, mutation_ops, selection_ops, verbose=True)


# In[539]:


columns = ['operators', 'strategy', 'successes', 's.avg.min', 's.avg.evals', 's.avg.gens', 'avg.evals', 'avg.min', 'avg.time']
aframe2 = report(ametrics2, columns_names=columns)
aframe2.sort_values(by=['successes', 'avg.min'], ascending=False)


# Ορισμένοι καλοί συνδυασμοί έχουν ήδη αρχίσει να κάνουν αισθητή την παρουσία τους, κάτι που μας επιτρέπει εν τέλει να αρχίσουμε την διαδικασία αποκλισμού γενετικών τελεστών. Το επόμενο μας βήμα θα είναι να αφαιρέσουμε αυτούς τους συνδυασμούς και να αρχίσουμε ξανά την αναζήτηση μας, μικραίνοντας ακόμη και κατά 3 τάξης μεγέθους το `delta`. Ωστόσο κρίνεται σκόπιμο να αναφέρουμε τις παρατηρήσεις που μας οδήγησαν σε αυτές τις επιλογές.
# 
# - cxBlendHigh οδηγεί κατά κύριο λίγο σε κακιές εποδόσεις.
# - Οι High τελεστές μετάλλαξης και επιλογής "ακυρώνουν" σε συνάρτηση με το Δ penalty αρκετούς απογόνους.
# - Low συνδυασμοί τελεστών ανεξαρτήτα του γενετικού αλγόριθμου σημειώνουν αρκετά καλή επίδοση.
# 
# Ο πρώτος και τρίτος λόγος δεν χρειάζονται κάποια περαιτέρω δικαιολόγηση για να πιστεί κανείς, αρκεί απλά να κοιτάξει τον πίνακα των αποτελεσμάτων. Όσο για την δεύτερη παρατήρηση μπορούμε μια ευκολία να παραθέσουμε μία βάσιμη αιτία που δικαιολογεί αυτήν την συμπεριφορά. Υψηλός τελεστής μετάλλαξης σημαίνει απαραίτητα και μεγάλη αλλαγή στο γενετικό υλικό, άρα και μεγαλύτερη πιθανότητα πολλοί απόγονοι να βγουν εκτός ορίων και επομένως να λάβουν πολύ μεγάλο penalty.

# In[546]:


kargs_cxBlend = {
    "function" : tools.cxBlend,
    "args" : {
        "alpha" : [1]
    }
}

kargs_cxSimulatedBinary = {
    "function" : tools.cxSimulatedBinary,
    "args" : {
        "eta" : [0.2]
    }
}

kargs_mutGaussian = {
    "function" : tools.mutGaussian,
    "args" : {
        "mu" : [0],
        "sigma" : [1],
        "indpb" : [0.5]
    }
}

kargs_mutPolynomialBounded = {
    "function" : mutPolynomialBounded,
    "args" : {
        "eta" : [0.2],
        "low" : [-5],
        "up" : [5],
        "indpb" : [0.5]
    } 
}

kargs_selTournament = {
    "function" : tools.selTournament,
    "args" : {
        "tournsize" : [5]
    }
}

ge_args = {
    "ge_with_stats" : None,
    "npop" : 100,
    "toolbox" : toolbox,
    "mu" : 100,
    "lambda_" : 200,
    "ngen" : 100,
    "cxpb" : 0.7,
    "mutpb" : 0.2,
    "rounds" : 5,
    "goal" : 0.0,
    "delta" : 0.000000000005,
    "verbose" : False
}


# In[544]:


ametrics3 = gridsearch(ge_args, strategy_ops, mate_ops, mutation_ops, selection_ops, verbose=True)


# In[545]:


aframe3 = report(ametrics3, columns_names=columns)
aframe3.sort_values(by=['successes', 'avg.min'], ascending=False)


# Παρόλο που για όλους του γενετικούς αλγόριθμους βρέθηκαν επιτυχημένοι συνδυασμοί τελεστών, θα πρέπει έναν απο αυτούς να διαλέξουμε για την τελική μας επιλογή. Η τελική επιλογή θα κριθεί τόσο από τα απόλυτα όσο και απο τα σχετικά κριτήρια. Θα ξεκινήσουμε βέβαια με τα πρώτα που αποτελούν και τα σημαντικότερα για την αξιολόγηση των γενετικών αλγορίθμων. Αν εξαιρέσουμε τις `avg.evals, avg.time` όπου εκεί εμφανίζεται έντονη διαφορά μεταξύ των αποτελεσμάτων, η `avg.time` που αποτελεί το πιο βασικό κριτήριο, έχει ακριβώς την ίδια τιμή σε όλα. Με τον χαρακτηρισμό "βασικό κριτήριο" δεν σημαίνει ότι οι υπόλοιπες είναι ήσσονος σημασίας αλλά οι αποκλίσεις που παρατηρούμε οφείλονται κυρίως στον τρόπο λειτουργίας των δύο αλγορίθμων. Η επιλογή μας θα καθοριστεί εν τέλει απο τα σχετικά κριτήρια. Όπως και στην περίπτωση των `avg.evals, avg.min` έτσι και οι `s.avg.evals, s.avg.min` οι τιμές είναι πρακτικά ίδιες. Η επιλογή μας θα καθοριστεί εν τέλει απο τα `s.avg.gens`. Από όλους τους συνδυασμούς εκείνος που σημειώσε τον μικρότερο αριθμό από γενιές για να φτάσει στην λύση είναι:
# 
# $$cxSimulatedBinaryLow,\ mutGaussianLow,\ selTournamentLow,\ eaMuPlusLambda\ 100\ 100\ 200$$

# In[551]:


ge_args = {
    "ge_with_stats" : None,
    "npop" : 100,
    "toolbox" : toolbox,
    "mu" : 100,
    "lambda_" : 200,
    "ngen" : 100,
    "cxpb" : 0.7,
    "mutpb" : 0.2,
    "rounds" : 10,
    "goal" : 0.0,
    "delta" : 0.0000000000005,
    "verbose" : False
}

strategy_ops = [eaMuPlusLambda_with_stats]
mate_ops = [kargs_cxSimulatedBinary]
mutation_ops = [kargs_mutGaussian]
selection_ops = [kargs_selTournament]


# In[552]:


ametrics4 = gridsearch(ge_args, strategy_ops, mate_ops, mutation_ops, selection_ops, verbose=True)


# In[553]:


aframe4 = report(ametrics4, columns_names=columns)
aframe4.sort_values(by=['successes', 'avg.min'], ascending=False)


# ### Για D=1, 10, 20, 40 και μεγαλύτερες διαστάσεις
# 
# Για τον βέλτιστο αλγόριθμο (που βρήκαμε για τις 2 διαστάσεις) και για διαφορετικές τιμές/τάξεις μεγέθους D=1, 10, 20, 40 ή και περισσότερων διαστάσεων του πεδίου ορισμού (σταθερά MAX_GENS και MAX_ROUNDS>=10) θα τυπώσουμε πίνακα με: **αριθμό διαστάσεων**, **αριθμό επιτυχιών**, **μέσο ολικό ελάχιστο**, **μέσο αριθμό αποτιμήσεων** και **μέσο χρόνο**.
# 
# Ευτυχώς το διαδικαστικό της υπόθεσης δεν θα μας δυσκολέψει. Έχουμε ήδη απο προηγούμενα ερωτήμα αναπτύξει ένα μεγάλο πλήθος παραμετροποιήσημων συναρτήσεων που και σε αυτή την περίπτωση λειτουργούν αποτελεσματικά. Ένας απλός wrapper, σχεδόν ίδιος με τον `prod_grisearch` θα δέχεται ως είσοδο την λίστα με την διαστάσεις και αφού λάβει τα αποτελέσματα από την evolution_with_stats, θα κρατάει εκείνα που ζητά η εκφώνηση για να εμφανίσει. Παρεά με την `dim_gridsearch` δημιουργήσαμε και την `new_toolbox` η οποία το μόνο που κάνει είναι να δημιουργεί κανούργια toolbox με βάση τις διαστάσεις που θέλουμε να ελέγχουμε κάθε φορά.

# In[593]:


def dim_gridsearch(ge_args, dims, verbose=False):
    metrics = []
    for dim in dims:
        ntoolbox = new_toolbox(dim)
        ge_args["toolbox"] = ntoolbox
        
        lmetrics = listify(evolution_with_stats(**ge_args))
        ret = [dim, lmetrics[0], *lmetrics[-3:]]

        if verbose:
            print(ret)

        metrics.append(ret)
    return metrics

def new_toolbox(dim):
    toolbox = base.Toolbox()
    toolbox.register("init_value", np.random.uniform, -35, 35)
    toolbox.register("individual", tools.initRepeat, creator.individual_container, toolbox.init_value, dim)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_ackley)
    toolbox.decorate("evaluate", tools.DeltaPenality (feasible, 6 * dim, distance))
    
    toolbox.register("mate", tools.cxSimulatedBinary, eta=0.2)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=1.0, indpb=0.5)
    toolbox.register("select", tools.selTournament, tournsize=5)
    
    return toolbox


# In[597]:


ge_args = {
    "ge_with_stats" : eaMuPlusLambda_with_stats,
    "npop" : 100,
    "toolbox" : None,
    "mu" : 100,
    "lambda_" : 200,
    "ngen" : 100,
    "cxpb" : 0.7,
    "mutpb" : 0.2,
    "rounds" : 15,
    "goal" : 0.0,
    "delta" : 0.000000000005,
    "verbose" : False
}


# In[594]:


dims = [1, 10, 20, 40]

dimf = dim_gridsearch(ge_args, dims, verbose=True)


# In[579]:


columns = ['dim', 'successes', 'avg.evals', 'avg.min', 'avg.time']
dimframe = report(dimf, columns_names=columns)
dimframe.sort_values(by=['successes', 'avg.min'], ascending=False)


# Ενώ τόσο στη μία όσο και στις δύο διαστάσεις ο γενετικός αλγόριθμος σημειώνει πλήρης επιτυχία, όσο οι διαστάσεις μεγαλώνουν το φαινόμενο αυτό σταματά. Ωστόσο το σημαντικό μέγεθος που πρέπει να μας αποσχολεί δεν είναι οι επιτυχίες αλλά το `avg.min`, καθώς μια προσεγγιστικά καλή λύση μπορεί να μην θεωρείται επιτυχημένη όταν το `delta` είναι αρκετά μεγαλό. Αν λοιπόν το ξεχάσουμε και επικεντρωθούμε στην βασική μας μετρική τα πράγματα έχουν ως εξής. Οι εξελικτικοί αλγόριθμοι όπως γνωρίζουμε ξεκινάν από ένα βασικό πληθυσμό ατόμων, τα όποια από γενιά σε γενία, περνάνε απο έναν κύκλο διασταυρώσεων και μεταλλάξεων παράγοντας νέους πιθανόν καλύτερους απογόνους. Μέσω της φυσικής επιλογής, εκείνη που έχουν περισσότερες πιθανότητες επιβιώσης, συνεχίζουν δημιουργώντας νέους απογόνους. Μπορεί ωστόσο οι συνθήκες του κόσμου (διαστάσεις) να αλλάζουν ραγδαία και ο πληθυσμός να μην αρκεί ή να μην προλάβει ποτέ να προσαρμοστεί, μέχρι εν τέλει να εκλείψει. 
# 
# ---
# 
# Ο λόγος που κάναμε αυτό τον παραλληλισμό είναι γιατί θέλαμε με έναν πιο έμμεσο τρόπο να αποκωδικοποιήσουμε το πρόβλημα και να το περάσουμε μέσα από ένα πρίσμα πιο ανθρώπινο και επιστημονικό. Όπως οι συνθήκες αλλάζουν, έτσι και οι διαστάσεις, απαιτείται μεγαλύτερος πληθυσμός άρα περισσότερες γενιές ώστε να "προλάβει" προσαρμοστεί και εν τέλει να επιδιώσει. Έτσι, ακόμα και όταν παράγονται άκυρα άτομα κατά την διαδικασία της διασταύρωσης, πολύ περισσότερο μάλιστα κατά την διαδικασία της μετάλλαξης, θα είναι πιο πιθανό κατά την φάση της επιλογής να βρεθούν εύκολα άτομα με καλή τιμή καταλληλότητας.

# 
# ### Βελτιστοποίηση σε μεγάλες διαστάσεις
# 

# Με τα παραπάνω κατά νου, επιχειρούμε αυξάνοντας το μέγεθος του πληθυσμού και των γενεών, να παίρνουμε όλο και καλύτερα αποτελέσματα. Ωστόσο πειδή ο γενετικός μας αλγόριθμος είναι τύπου `mu,lambda` πρέπει ανάλογα με τον αριθμό του πληθυσμού να αυξάνουμε και αυτά τα μεγέθη. Σε πρώτο βήμα θα τρέξουμε ξάνα και για τις τέσσερις διαστάσεις για να βεβαιωθούμε ότι η ιδέα μας αποδεικνύεται σωστή

# In[595]:


ge_args["npop"] = 400
ge_args["ngen"] = 400
ge_args["mu"] = 300
ge_args["lambda_"] = 500


# In[ ]:


dimf2 = dim_gridsearch(ge_args, dims)


# In[629]:


dimframe2 = report(dimf2, columns_names=columns)
dimframe2.sort_values(by=['successes', 'avg.min'], ascending=False)


# Έχοντας πλέον σταθερή διάσταση 40, θα συνεχίσουμε στην ίδια λογική, προσπαθώντας να παίρνουμε όλο και καλύτερα αποτελέσματα. 

# In[598]:


ge_args["npop"] = 800
ge_args["ngen"] = 400
ge_args["mu"] = 300
ge_args["lambda_"] = 500

dims = [40]


# In[600]:


dimf3 = dim_gridsearch(ge_args, dims, verbose=True)


# In[601]:


ge_args["npop"] = 800
ge_args["ngen"] = 800
ge_args["mu"] = 300
ge_args["lambda_"] = 500


# In[602]:


dimf4 = dim_gridsearch(ge_args, dims, verbose=True)


# In[605]:


ge_args["npop"] = 1200
ge_args["ngen"] = 800
ge_args["mu"] = 300
ge_args["lambda_"] = 500


# In[606]:


dimf5 = dim_gridsearch(ge_args, dims, verbose=True)


# In[634]:


dim_metris = [dimf3[0], dimf4[0], dimf5[0]]
report(dim_metris, columns_names=columns)


# Πράγματι, όπως βλέπουμε στον πίνακα που ακολουθεί, η μέση τιμή `avg.min` των καλύτερων εκτιμήσεων μειώνεται όλο και περισσότερο, προσεγγίζοντας όλο και καλύτερα την πραγματική τιμή 0. Παρατηρούμε ότι στην τελευταία μας δοκιμή παρόλο την αύξηση του πληθυσμού και των γενεών μικρότερη βελτίωση συγκριτική με την προηγούμενη. Δεν θα έπρεπε ωστόσο να μας παραξενέυει, καθώς δεν θα μπορούσαμε να περιμένουμε το σύστημα να βελτιστοποιείται συνέχεια και με τον ίδιο ρυθμό.
# 
# 
# ---- 
# 
# Τέλος το μόνο που μένει είναι να βρούμε μια νέα -σταθερή- διάσταση, πιθανότατα μικρότερη από την προηγούμενη και ένα `delta` που να σας δίνουν 35% - 50% επιτυχίες. Αυτό μπορούμε να το επιτύχουμε μόνο εμπειρικά, έχοντας στο μυαλό μας πως προφανώς το `delta` θα πρέπει να μεγαλώνει όσο αυξάνει η διαστατικότητα. Ύστερα απο **αρκετές** δοκιμές βρήκαμε τον συνδυασμό των παραμέτρων που το πετυχαίνει.

# In[647]:


ge_args["npop"] = 1200
ge_args["ngen"] = 800
ge_args["mu"] = 300
ge_args["lambda_"] = 500
ge_args["delta"] = 0.5


# In[648]:


final1 = dim_gridsearch(ge_args, [30], verbose=True)


# Μειώνοντας το `delta` στο μισό (σε 0.25) έχουμε την δυνατότητα να υπερδιπλασιάσουμε το ποσοστό των επιτυχιών. Ωστόσο για να συμβεί αυτό έπρεπε να αυξήσουμε των αριθμό του πληθυσμού και των γενεών κατά ένα κατά τον μισό του παράγοντα αλλαγής του `delta`.

# In[ ]:


ge_args["npop"] = 1600
ge_args["ngen"] = 100
ge_args["mu"] = 300
ge_args["lambda_"] = 500
ge_args["delta"] = 0.25


# In[661]:


final2 =  dim_gridsearch(ge_args, [30], verbose=True)


# Δυστυχώς αν ξαναμειώσουμε το `delta` στο μισό με την ίδια αλλαγή στα άλλα 2 μεγέθη, το αποτέλεσμα δεν θα είναι το ίδιο (δεν γίνεται και πρακτικά μιας και το μέγιστο είναι 50). Θα πρέπει να διπλασιάσουμε σχεδόν και τα άλλα δύο μεγέθη για να καταφέρουμε και τον διπλασιασμό των επιτυχίων.
