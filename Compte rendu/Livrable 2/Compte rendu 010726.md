Aujourd'hui, nous avons testé plusieurs fonctions telles que :



* **robot.set\_jog\_control\_mode** : Active un mode spécial pour que le robot bouge en continu sans s'arrêter.
* **with robot.jog\_control()** : Prévient le robot qu'on va lui envoyer plein d'ordres très vite.
* **robot.jog** : Envoie les mouvements en direct sans bloquer le code pour éviter les saccades.
* **wait\_for\_execution=False** : Dit à la caméra de continuer à filmer sans attendre que le robot ait fini son geste.



Pour rendre le mouvement du robot plus fluide. Cependant, d'après plusieurs tests, nous avons constaté que les fonctions trouvées appartiennent toutes à une version récente ce qui rend impossible l'interaction avec le robot Niryo Ned qui possède un système plus ancien.



Nous allons continuer à chercher la solution, notamment en modifiant nous-mêmes la taille des pas de déplacement pour optimiser la fluidité du mouvement.

