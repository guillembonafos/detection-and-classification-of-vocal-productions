import scripts.functions.preprocessing as pp
from scripts.functions.metadata import metadata_longform_baby
from scripts.functions.hypermodel import WinWavTransferLearning
from scripts.functions.segmentation import segmentation, df_pred, wav_creation
import os
import tensorflow as tf
import kerastuner as kt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sklearn as sk
from datetime import datetime
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

#####################
# Load the metadata #
#####################

# Metadata
meta_continu = metadata_longform_baby()
# If we need the length of the files. Much longer!
meta_continu = metadata_longform_baby(length=True)
# We save to do not rerun the function later
meta_continu.to_csv(os.path.join(os.getcwd(), 'data/meta_continu_bebe.csv'))
# Direct loading
meta_continu = pd.read_csv(os.path.join(os.getcwd(), 'data/meta_continu_bebe.csv'))

##################################
# Prédiction des enregistrements #
##################################

ds = pp.preparation_longform_baby(meta_continu, batch_size=32, transfer_learning=True)
input_shape = next(iter(ds.unbatch()))[0].shape[0]

# Récupération du modèle
hm = WinWavTransferLearning(input_shape=input_shape, n_labels=5)
hp = kt.HyperParameters()
tuner = kt.tuners.bayesian.BayesianOptimization(hypermodel=hm,
                                                hyperparameters=hp,
                                                objective=kt.Objective("val_loss", direction="min"),
                                                max_trials=1,
                                                num_initial_points=1,
                                                tune_new_entries=True,
                                                overwrite=False,
                                                project_name='babblecor')

modele = tuner.get_best_models()[0]
modele.summary()

# La première itération prend beaucoup de temps. Puis, très rapide.
start = datetime.now()
y = modele.predict(ds)
duree_pred = datetime.now() - start
print(duree_pred, "seconds")

# Autre version, pour retrouver les infos de l'audio que l'on prédit.
#pred = []
#start = datetime.now()
#for batch in ds:
#    x, date, enfant = batch
#    yhat = modele.predict(x)
#    pred += [yhat, date.numpy(), enfant.numpy()]
#duree_pred = datetime.now() - start
#print(duree_pred, "seconds")


# On enregistre les résultats
with open("donnees/pred_bebe.pickle", "wb") as fp:
    pickle.dump(y, fp)

# Pour les recharger si besoin
with open("donnees/pred_bebe.pickle", 'rb') as handle:
    y = pickle.load(handle)

#######################################################################
# La création des fichiers reprenant l'information quant au moment de #
# vocalisation dans les enregistrements continus                      #
#######################################################################

# On écrit les prédictions sur des fichiers txt, pour chaque enregistrement
segmentation(meta_continu, y, baby=True)

# On crée les fichiers wav
wav_creation(fd="/home/guillhem/pCloudDrive/Documents/babyvoc/enregistrements",
             baby=True)

##################################################
# Travail exploratoire sur la segmentation faite #
##################################################

# On ajoute la variable seg_file, le fichier txt reprenant la segmentation faite,
# afin d'avoir toute l'information nécessaire directement dans le df
meta_continu["seg_file"] = meta_continu["files"].map(lambda x: str(x)[:-4] + ".txt")

# On prend les informations nécessaires dans un df à partir des prédictions
df = df_pred(meta_continu, y, bebe=True)

# On enregistre pour ne pas avoir à le relancer une autre fois
df.to_csv(os.path.join(os.getcwd(), 'donnees/df_composition_bebe.csv'))

# Temps total d'enregistrement en secondes
tps_sec = meta_continu["len"].sum()/16000
# Temps en minutes
tps_min = tps_sec/60
# Temps en heures
tps_h = tps_min/60
print("Les enregistrements à classifier durent", tps_sec, "secondes, soit", tps_min, "minutes ou encore", tps_h, "heure")

# On récupère le nom des codes de classe
# chemin des fichiers
fd = "/home/guillhem/Documents/babyvoc/donnees_babblecor"
# Import des métadonnées à disposition
metadonnees = pd.read_csv(os.path.join(fd, "private_metadata.csv"))
# Renommage des colonnes pour coller aux noms données dans les autres fonctions
metadonnees.rename(columns={"Answer": "label", "clip_ID": "fd"}, inplace=True)
metadonnees.dropna(inplace=True)

le = sk.preprocessing.LabelEncoder()
le.fit(metadonnees["label"])
le.classes_
df["classe"] = le.inverse_transform(df["type_voc"].astype(int))
df["classe"] = df["classe"].astype("str")

#############################
# Statistiques descriptives #
#############################

# Secondes de voc pour chaque mois d'enregistrement
df.groupby("enfant")["n_records"].value_counts()
# Nombre de vocs par classe
df.groupby("enfant")["classe"].value_counts()
# Secondes de vocalisation à chaque mois
df.groupby("n_records").count()["voc"].mean()
# Variance
df.groupby("n_records").count()["voc"].var()
# écart-type
np.sqrt(df.groupby("n_records").count()["voc"].var())
# Par enfant,
df.groupby(["enfant", "n_records"]).count()["voc"].mean()
np.sqrt(df.groupby(["enfant", "n_records"]).count()["voc"].var())
# Par voc
voc_months = df.groupby(["n_records", "type_voc"]).count()["voc"].unstack()
voc_months["n_records"] = voc_months.index.astype(int)
voc_months.rename(columns={0: "Canonical", 1: "Crying", 2: "Junk", 3: "Laughing", 4: "Non-canonical"}, inplace=True)
voc_months = pd.melt(voc_months, id_vars="n_records", value_vars=["Canonical", "Crying", "Junk", "Laughing", "Non-canonical"])
voc_months.fillna(0, inplace=True)
voc_months.groupby(["type_voc", "n_records"])["value"].sum().groupby("type_voc").mean()
np.sqrt(voc_months.groupby(["type_voc", "n_records"])["value"].sum().groupby("type_voc").var())
# Détail pour chaque enfant
df.groupby(["enfant", "n_records"]).size().groupby("enfant").mean()
np.sqrt(df.groupby(["enfant", "n_records"]).size().groupby("enfant").var())
# Moyennes, variance et écart-type pour chaque vocalisation
# Sur l'année
df.groupby("n_records")["classe"].value_counts().groupby("classe").mean()
np.sqrt(df.groupby("n_records")["classe"].value_counts().groupby("classe").var())
# Par mois
df.groupby("n_records")["classe"].value_counts()
# Somme sur l'année, pour chaque classe
df.groupby("classe").sum()["voc"]
# Moyenne par mois
df.groupby("classe")["voc"].value_counts() / 11

#########################
# Vocs pour chaque mois #
#########################

sns.set()

fig, ax = plt.subplots()
ax = sns.displot(data=df, x="n_records", discrete=True, kde=True, hue="classe", multiple="dodge",
            height=10, aspect=1, facet_kws={"legend_out": True})
plt.xticks(rotation=45)
#plt.legend(bbox_to_anchor=(1.1, 0.8), loc=2)
ax.set_axis_labels("Month", "Second of vocalization")
plt.savefig(os.path.join(os.getcwd(), "images/distributions_vocs_ensemble_babblecor.png"), bbox_inches="tight")
plt.show()

plt.figure(figsize=(15, 10))
g = sns.FacetGrid(df, col="classe", col_wrap=3, sharex=True, sharey=True)
g.map_dataframe(sns.histplot, x="n_records", stat="probability", kde=True)
g.set_axis_labels("Month", "Number of vocalization")
g.set_titles(col_template="{col_name}")
g.set_xticklabels(rotation=45)
plt.savefig(os.path.join(os.getcwd(), "images/distributions_vocs_babblecor.png"), bbox_inches="tight")
plt.show()

voc_months = df.groupby(["n_records", "type_voc"]).count()["voc"].unstack()
voc_months["n_records"] = voc_months.index.astype(int)
voc_months.rename(columns={0: "Canonical", 1: "Crying", 2: "Junk", 3: "Laughing", 4: "Non-canonical"}, inplace=True)

voc_months = pd.melt(voc_months, id_vars="n_records", value_vars=["Canonical", "Crying", "Junk", "Laughing", "Non-canonical"])
voc_months.fillna(0, inplace=True)

plt.figure(figsize=(15, 10))
sns.lineplot(data=voc_months, x="n_records", y="value", hue="type_voc")
plt.xlabel("Month")
plt.ylabel("Seconds of vocalization")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig(os.path.join(os.getcwd(), "images/evolution_vocs_babblecor.png"), bbox_inches="tight")
plt.show()

voc_days = df.groupby([df.index, "classe"]).count()["voc"].unstack()
voc_days.fillna(0, inplace=True)
voc_days["day"] = voc_days.index
voc_days = pd.melt(voc_days, id_vars="day", value_vars=["Canonical", "Crying", "Junk", "Laughing", "Non-canonical"])

fig, ax = plt.subplots()
ax = sns.displot(data=voc_days, x="day", discrete=True, kde=True, hue="classe", multiple="dodge",
            height=10, aspect=1, facet_kws={"legend_out": True})
plt.xticks(rotation=45)
#plt.legend(bbox_to_anchor=(1.1, 0.8), loc=2)
ax.set_axis_labels("Month", "Second of vocalization")
plt.savefig(os.path.join(os.getcwd(), "images/distributions_vocs_ensemble_babblecor.png"), bbox_inches="tight")
plt.show()

#
# Graphs pour ilcb #
####################

df["n_records"] = df["n_records"].astype(str)

df_sub = df[~(df["classe"].isin(["Canonical", "Junk"]))]
df_sub = df_sub.assign(child=np.where(df_sub["enfant"] == "eva", 0, 1))

plt.figure(figsize=(15, 10))
g = sns.FacetGrid(df_sub, col="classe", row="child", sharex=True, sharey=True)
g.map_dataframe(sns.histplot, x="n_records", stat="probability", kde=True)
g.set_axis_labels("Month", "Vocalization (normalized)")
g.set_titles(col_template="{col_name}")
g.set_xticklabels(rotation=45)
plt.savefig(os.path.join(os.getcwd(), "images/distributions_vocs_babblecor_enfant.png"), bbox_inches="tight")
plt.show()
