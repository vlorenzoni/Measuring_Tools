# Measuring_Scripts

A suite of Python scripts designed for acoustic measurements, sweep generation, and Room Impulse Response (RIR) analysis. This repository provides a standalone workflow for capturing and processing acoustic data without requiring a Digital Audio Workstation (DAW).

## 🚀 Key Features

* **Measurement Suite**: Scripts for generating logarithmic sweeps and synchronized play/record functionality.
* **Acoustic Analysis**: Automated calculation of acoustic parameters including reverberation time (RT), EDT, and octave-band analysis.
* **Calibration Tools**: Specialized scripts for microphone and probe calibration to ensure measurement accuracy[cite: 3].
* **Library Modules**: Custom modules for acoustic feedback simulation and signal processing[cite: 3].

## 📂 Repository Structure

- **`Measuring/`**: Core measurement scripts and hardware interface logic[cite: 3].
- **`Processing/`**: Tools for analyzing captured recordings and calculating RIR parameters[cite: 3].
- **`lib/`**: Shared acoustic utilities and signal processing algorithms[cite: 3].
- **`Recordings/` & `RIRs/`**: Local directories for data storage (configured in `.gitignore`)[cite: 3].

## Files in Measuring folder
- **`generateSweep_measure_calculateRIR--singleMic.py/`**: Creates sweep play and record adn evaluate and stores RIR for one specified microphone.
- **`generateSweep_measure_calculateRIR--PROBE.py`**: Creates sweep play and record adn evaluate and stores RIRs for all probe microphones.
- **`calibration/`**: Used to evaluate the system audio delay between sending and recording the signal. Need to shortcut the system before measuring.


## 🛠 Setup & Requirements

Ensure you have Python 3.10+ installed. Install the necessary processing libraries[cite: 3]:


```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
