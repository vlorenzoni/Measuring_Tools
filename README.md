# Measuring_tests

A suite of Python scripts designed for acoustic measurements, sweep generation, and Room Impulse Response (RIR) analysis. This repository provides a standalone workflow for capturing and processing acoustic data without requiring a Digital Audio Workstation (DAW).

## 🚀 Key Features

* **Measurement Suite**: Scripts for generating logarithmic sweeps and synchronized play/record functionality.
* **Acoustic Analysis**: Automated calculation of acoustic parameters including reverberation time (RT), EDT, and octave-band analysis.
* **Calibration Tools**: Specialized scripts for microphone and probe calibration to ensure measurement accuracy[cite: 3].
* **Library Modules**: Custom modules for acoustic feedback simulation and signal processing[cite: 3].

## 📂 Repository Structure

- **`Measure_noDAW/`**: Core measurement scripts and hardware interface logic[cite: 3].
- **`Processing/`**: Tools for analyzing captured recordings and calculating RIR parameters[cite: 3].
- **`lib/`**: Shared acoustic utilities and signal processing algorithms[cite: 3].
- **`Recordings/` & `RIRs/`**: Local directories for data storage (configured in `.gitignore`)[cite: 3].

## 🛠 Setup & Requirements

Ensure you have Python 3.10+ installed. Install the necessary processing libraries[cite: 3]:

```bash
pip install numpy scipy matplotlib sounddevice
