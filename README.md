# GNC Bench : banc d'essai GNC-in-the-loop (MEC McGill)

Simulateur de quadricoptère en 6 degrés de liberté, dans lequel on branche un contrôleur pour le tester automatiquement. Ce dépôt contient le code de la **semaine 1** : dynamique, intégrateur RK4 et contrôleur de vol stationnaire, avec 7 tests qui passent.

---

## 1. Installer les logiciels (une seule fois)

Tout se fait dans **VS Code** avec **Python**. Pas besoin de MATLAB.

| Logiciel | Où le trouver | Remarque |
| --- | --- | --- |
| VS Code | déjà installé | |
| Python 3.12 | [python.org/downloads](https://www.python.org/downloads/) | Windows : cocher **« Add python.exe to PATH »** pendant l'installation |
| Git | [git-scm.com](https://git-scm.com/downloads) | Mac : déjà présent, ou `xcode-select --install` |
| Compte GitHub | [github.com](https://github.com) | pour sauvegarder le code et le montrer au jury |

Vérifie dans un terminal :

```bash
python --version     # Mac/Linux : python3 --version
git --version
```

## 2. Ouvrir le projet dans VS Code

1. Dézippe `gnc-bench.zip` dans un dossier simple, par exemple `Documents/gnc-bench`.
2. Dans VS Code : **File > Open Folder…** et choisis `gnc-bench`.
3. VS Code propose d'installer les extensions recommandées (Python, Pylance, Ruff, Jupyter). Clique **Install All**.

## 3. Créer l'environnement Python

Un environnement virtuel (`.venv`) isole les librairies de ce projet. Ouvre le terminal intégré (**Terminal > New Terminal**), puis :

**Mac / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Si PowerShell refuse d'activer : `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, puis recommence.

Ensuite, dans VS Code : `Ctrl+Shift+P` (Mac : `Cmd+Shift+P`) > **Python: Select Interpreter** > choisis celui qui contient `.venv`. Le préfixe `(.venv)` doit apparaître dans le terminal.

## 4. Vérifier que tout marche

```bash
pytest
```

Résultat attendu : `7 passed`. Tu peux aussi ouvrir l'onglet **Testing** (icône d'éprouvette) dans VS Code et cliquer sur ▶.

Puis lance la démo, avec **F5** (configuration « Démo vol stationnaire ») ou :

```bash
python scripts/hover_demo.py
```

Le drone part incliné du sol et rejoint le point (1 ; -0,5 ; 2) m avec un lacet de 0,5 rad. Résultat attendu : erreur finale d'environ 0,6 cm, et un graphique dans `outputs/hover_demo.png`.

## 5. Mettre le projet sur GitHub

1. Sur github.com : **New repository**, nom `gnc-bench`, sans README.
2. Dans le terminal du projet :

```bash
git init
git add .
git commit -m "Semaine 1 : dynamique 6 DDL, RK4, contrôleur de vol stationnaire"
git branch -M main
git remote add origin https://github.com/<ton-nom>/gnc-bench.git
git push -u origin main
```

Ensuite, fais un commit à chaque étape qui marche. VS Code le fait aussi en un clic dans l'onglet **Source Control**.

## 6. Ce que contient le code

```
gnc-bench/
├── src/gncbench/
│   ├── params.py        paramètres physiques (masse, inertie, bras, rotors) et matrice de mélange
│   ├── math3d.py        quaternions, matrices de rotation, angles d'Euler
│   ├── dynamics.py      équations du mouvement 6 DDL : dx/dt = f(x, u)
│   ├── integrators.py   pas de Runge-Kutta 4
│   ├── controllers.py   interface Controller + contrôleur de vol stationnaire
│   └── sim.py           boucle de simulation (physique 500 Hz, contrôleur 250 Hz)
├── scripts/hover_demo.py  démo + graphique
├── tests/test_dynamics.py 7 tests de validation
└── .vscode/             réglages VS Code (interpréteur, tests, F5)
```

**Le modèle.** L'état a 13 composantes : position et vitesse dans le repère monde (z vers le haut), attitude en quaternion, vitesses angulaires dans le repère corps. Les équations :

- ṗ = v
- m v̇ = R(q) [0, 0, T] − m g e₃ − c v
- q̇ = ½ q ⊗ [0, ω]
- J ω̇ = τ − ω × J ω

**Le contrôleur.** Une boucle externe PID calcule l'accélération voulue. On en déduit l'orientation voulue du drone, puis un contrôle géométrique d'attitude sur SO(3) (Lee, Leok et McClamroch, 2010) calcule les couples. Une matrice de mélange les convertit en poussée pour chacun des 4 rotors.

**Les tests** vérifient la physique, pas seulement que le code tourne : chute libre comparée à la solution analytique, équilibre exact à la poussée de vol stationnaire, conservation de l'énergie et du moment cinétique en rotation libre, et convergence du contrôleur.

## 7. Ton travail pour la semaine 1

Le code tourne, mais l'objectif est que tu puisses l'expliquer au jury. Dans l'ordre :

- [ ] Lire `dynamics.py` et retrouver chaque équation ci-dessus dans le code.
- [ ] Changer la masse à 1,5 kg dans `params.py`, relancer la démo, et expliquer ce qui change sur les poussées.
- [ ] Mettre `ki=0` puis `ki=1.0` dans `HoverController` et comparer les courbes. Pourquoi l'intégrale ralentit-elle la convergence finale ?
- [ ] **À coder toi-même** : ajouter une dynamique de moteur du premier ordre (la poussée réelle suit la commande avec une constante de temps d'environ 30 ms). Indice : ajouter 4 composantes à l'état.
- [ ] Ajouter un test pour ta dynamique de moteur.
- [ ] Faire un commit sur GitHub.

La semaine 2 ajoutera les capteurs bruités, le vent et les pannes.
