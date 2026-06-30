Aujourd'hui, nous avons réussi à contrôler le mouvement de tous les 6 axes du robot et nous avons pu attraper un objet (un petit dé en mousse).



Pour faire ça, nous avons appliqué le contrôle articulaire continu. C’est du suivi en temps réel sans enregistrer les positions à l'avance dans le code. C'est à dire qu'au lieu de dire au robot : "Va à telle position fixe", le programme lui dit en boucle : "Ajoute un tout petit angle à tes moteurs dans cette direction". Plus précisément, on récupère la position actuelle du robot avec la fonction robot.get\_joints() et on ajoute ou soustrait une petite valeur selon le geste de notre main.



Voici les détails :  



1. Lire la position actuelle : angles\_actuels = robot.get\_joints()
2. Modifier l'axe 1 (la base) : angles\_actuels\[0] = angles\_actuels\[0] + 0.12 (on tourne un tout petit peu vers la gauche)
3. Envoyer la nouvelle position : robot.move\_joints(angles\_actuels)



Ensuite, nous avons défini nous-mêmes 16 gestes de mains et leurs actions. Le but était que chaque geste ressemble visuellement au mouvement physique du robot pour que ce soit facile et intuitif à contrôler pour l'utilisateur.



Liste des gestes et des actions :



1\. Gestes uniques (enregistré les coordonnées à l'avance dans le code)



* Paume contre paume doigts vers Le haut : Retour à l'origine verticale (tous les axes à 0).
* Mains levées : Retour à la position de départ après calibration.
* Main levée : Ouvrir la pince.
* Poing levée : Fermer la pince.



2\. Gestes continus



* Main avec index pointant à gauche : Tourner la base à gauche (Axe 1).
* Main avec index pointant à droite : Tourner la base à droite (Axe 1).
* Index vers le haut : Lever / reculer le bras (Axe 2).
* Index vers l'utilisateur : Baisser / avancer le bras (Axe 2).
* Pouce vers le haut : Plier le coude vers le haut (Axe 3).
* Pouce vers le bas : Déplier le coude vers le bas (Axe 3).
* Main vers la gauche : Tourner le poignet à gauche (Axe 4).
* Main vers la droite : Tourner le poignet à droite (Axe 4).
* Paume vers le haut : Pencher la pince vers le haut (Axe 5).Paume vers le bas : Pencher la pince vers le bas (Axe 5).
* Pouce et index rapprochés : Rotation horaire de la pince (Axe 6).
* Bout des doigts joints : Rotation anti-horaire de la pince (Axe 6).



Cependant, d'après nos tests, l'objet reste difficile à attraper et cela prend du temps car il faut changer plusieurs fois de geste pour chaque axe. Nous allons donc essayer d'améliorer le programme pour rendre les mouvements plus fluides et le robot plus facile à contrôler.

