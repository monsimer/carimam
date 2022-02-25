# CARIMAM

Classification de signaux de mammifères marins dans les Caraïbes

## Pré-traitement

2 types de pré-traitement : 
- spectral_extraction : permet de sortir les spectrogrammes des signaux
- scalo_DB : permet de sortir les scalogrammes des signaux, par ondelettes de daubechies

## Clustering

2 méthodes testées :
- CNN : utilisation d'un réseau de neurones déjà entrainé sur une autre zone géographique
- sort_cluster : clustering avec HDBCSAN
