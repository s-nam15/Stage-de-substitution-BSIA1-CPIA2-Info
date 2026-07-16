1. ##### Bibliothèques à installer 



Installation des outils de traitement de données et Machine Learning :



* pip install mediapipe==0.10.9 opencv-python 



*Si le programme ne se lance pas à cause d'une erreur liée au module **« solutions »** de MediaPipe, installez une version compatible qui contient le module <b>« solutions »</b>* *de MediaPipe (comme la 0.10.9) en demandant simplement à une IA la commande d'installation adaptée à votre système.*



Installation des outils de traitement de données et Machine Learning :



* pip install pandas numpy scikit-learn joblib







##### 2\. Guide d'utilisation 



1. **Pour tester le système :** Activez la caméra ou la webcam de votre PC, puis exécutez le script Python nommé **« test\_model.py »**.
2. **Pour enrichir la base de données et réentraîner le modèle :** Exécutez d'abord le script **« collect\_data.py »** pour enregistrer les coordonnées de vos mains dans le fichier **« test.csv »**. Lancez ensuite **« train\_model.py »** pour entraîner le modèle avec ces nouvelles données.



*Pour plus de détails sur le fonctionnement, vous pouvez consulter les diagrammes de séquence dans la partie 3 (Conception) de mon rapport du livrable 1.*



