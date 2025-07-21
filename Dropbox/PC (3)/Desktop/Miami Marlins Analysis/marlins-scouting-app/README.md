# ⚾ AI-Powered Baseball At-Bat Simulator

This project simulates a Major League Baseball at-bat using machine learning models trained on Statcast data. Coaches, analysts, and fans can select a pitcher, configure a pitch type and location, and simulate how a specific hitter would respond — including swing probability, whiff/contact prediction, and expected batted ball outcomes (coming soon).

The system is built with a multi-model pipeline and a web interface that provides intuitive visualization of each simulated pitch scenario.

---

## Key Components & File Descriptions

### Model Training

- [`train_sequential_models.py`](./train_sequential_models.py)
  Trains the *Swing vs No Swing* classification model. This model predicts whether a batter will swing at a given pitch based on pitch characteristics, count context, and batter-specific tendencies.

- **`train_whiff_vs_contact_model.py`**  
  Trains the *Whiff vs Contact* model. If the batter swings, this model determines whether the swing results in a whiff (miss) or contact (hit/ball in play).

---

### Simulation Engine

- **`at_bat_simulation.py`**  
  The core script that drives the simulation. It loads the trained models and uses them in sequence to simulate a pitch result:  
    1. Will the batter swing?  
    2. If yes, is it a whiff or contact?  
    3. If contact, ends and resets at bat
    4. Walk and strikeout events end at bat as well 

---

### Backend API

- **`pybaseball_service.py`**  
  This Python backend server handles API requests from the frontend React app. It receives pitch input parameters, runs the simulation using the above models, and returns the result to be displayed.

---

### Data Downloading

- **`download_acuna_complete_career.py`**  
  Downloads complete career pitch-level Statcast data for Ronald Acuña Jr. Modify the player ID in this file to fetch data for any other MLB player. This script powers both hitter and pitcher model training workflows.

---

### Frontend Interface (React)

- **`page.tsx`**  
  Located at:  
  `Marlins-Predictive-Simulation/Dropbox/PC (3)/Desktop/Miami Marlins Analysis/marlins-scouting-app/src/app/simulate-atbat/page.tsx`  
  This is the main frontend page for the at-bat simulator. It allows the user to:  
    - Select a pitcher and opposing hitter  
    - Choose pitch type and location  
    - Submit inputs to simulate the at-bat and display the results with contextual insights

## Tech Stack

- **Frontend**: React, TypeScript  
- **Backend**: Python (FastAPI or Flask style)  
- **ML Models**: XGBoost, scikit-learn, Transformers  
- **Data**: MLB Statcast via pybaseball  

---


