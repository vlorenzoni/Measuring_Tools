# %% Import libraries
from lib.py_octave_band import octavefilter
from lib import acoustics_original
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import soundfile as sf
import os
from pathlib import Path
import scienceplots
plt.style.use(["science", "ieee", "grid", "bright"])
# %matplotlib widget

# %% Define the class Measurement
class Measurement:
    def __init__(self, folder_name, measurement_prefix, microphone_identifier, fs):
        self.param_labels = [
            "$T_{20}$ [s]", "EDT [s]", "$C_{80}$ [dB]", "$D_{50}$ [dB]", "$T_s$ [s]", "$E$ [dB]", "DRR"]
        self.folder_name = folder_name
        self.meas_name = measurement_prefix
        self.microphone_identifier = microphone_identifier
        self.fs = fs
        self.freqbands = None
        self.broadband_params = None
        self.subband_params = None
        self.rirs = []

    def calc_obj_params(self, octave_band=1, f_limits=[125, 8000]) -> None:
        """
        Calculate objective parameters for a given measurement.
        Parameters are calculated for the broadband signal and for octave bands.
        """
        def calculator(rir, fs):
            rt_20 = acoustics_original.rt(rir, fs, 60, db_start=5, db_end=25)
            edt = acoustics_original.edt(rir, fs)
            C80 = acoustics_original.clarity(rir, fs, 0.080)
            D50 = acoustics_original.definition(rir, fs, 0.050)
            Ts = acoustics_original.center_time(rir, fs)
            E = 10 * np.log10(np.sum(rir ** 2))
            drr = acoustics_original.direct_to_reverberant(rir, fs)
            return rt_20, edt, C80, D50, Ts, E, drr

        broadband_params = []
        subband_params = []

        for rir in self.rirs:
            # Calculate broadband parameters
            broadband_params.append(calculator(rir, self.fs))
            # Divide RIR into 1/3 octave bands
            _, freq, sigbands = octavefilter(
                rir, self.fs, octave_band, order=16, limits=f_limits)
            # Calculate parameters in every octave band
            subband_params.append([calculator(sig, self.fs) for sig in sigbands])

        self.freqbands = freq
        self.broadband_params = np.array(broadband_params).T
        self.subband_params = np.array(subband_params).T


def load_measurements(folder, measurement_prefix, microphone_identifier):
    measurements = []

    for i in measurement_prefix:
        # Find the subfolder for each measurement prefix
        subfolder = Path(folder) / f"{i}"
        print(f"Processing folder: {subfolder}")

        if not subfolder.exists():
            print(f"Folder {subfolder} does not exist.")
            continue

        rirs = []
        fs = None  # Initialize sampling rate

        for file in os.listdir(subfolder):
            # Only pick files containing the microphone_identifier
            if any(mic in file for mic in microphone_identifier):
                # Load the file
                rir, fs = sf.read(subfolder / file)

                print("read file {}/{}".format(subfolder , file))
                # concatenate strings subforlder and file
                rirs.append(rir)

        # Skip if no valid RIRs were found
        if not rirs:
            print(f"No valid RIRs found in {subfolder}")
            continue

        # Create a Measurement object and calculate parameters
        measurement = Measurement(subfolder, measurement_prefix, microphone_identifier, fs)
        measurement.rirs = rirs
        measurement.calc_obj_params()
        measurements.append(measurement)

    return measurements



## USER DEFINED PARAMETERS 

# %% Load configuration


folder = Path("/Users/apple/Library/CloudStorage/Dropbox/My_ESAT/Experiments/Measuring_tests/RIRs/")
measurement_prefix = ["pos_test_probe","pos_test_mic"] # ["A1","A2","A3","B1","B2","B3","C1","D2"] # indicate positions here
microphone_identifier = ["E"] # top microphone of the probe

fs = 48_000


f_limits = [100, 8_000]
octave_band = 1  # 1 octave

#check if the folder exists
if not folder.exists():
    print(f"Folder {folder} does not exist.")
    exit()


# %% Load measurements from microphone E
measurements = load_measurements(folder,measurement_prefix,microphone_identifier)




# %% Plot the objective parameters
titles = ["Reverberation Time ($T_{20}$)", "Early Decay Time (EDT)",
          "Clarity ($C_{80}$)", "Definition ($D_{50}$)", "Center Time ($T_s$)", "Energy ($E$)"]

matplotlib.use("pgf")
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

for mnr,meas in enumerate(measurements):
    # Make a big plot for six objective parameters
    fig, axs = plt.subplots(3, 2,figsize=(8, 10)) #figsize=(8, 10))
    freq = [125, 250, 500, 1000, 2000, 4000, 8000]
    for idx, param in enumerate(meas.subband_params[:6]):
        average_param = np.mean(param, axis=1)
        std_param = np.std(param, axis=1)
        axs[idx // 2, idx % 2].semilogx(freq, average_param, label=meas.meas_name, marker="s")
        axs[idx // 2, idx % 2].fill_between(freq, average_param - std_param,
                                            average_param + std_param, alpha=0.7)

        axs[idx // 2, idx % 2].set_xticks(freq, ["125", "250", "500", "1k", "2k", "4k", "8k"])
        axs[idx // 2, idx % 2].set_xlabel("Frequency [Hz]")
        axs[idx // 2, idx % 2].set_title(titles[idx])
        axs[idx // 2, idx % 2].set_ylabel(meas.param_labels[idx])
        # axs[idx // 2, idx % 2].legend()
    

    fig.set_tight_layout(True)
    text_width = 3.48761
    fig.set_size_inches(w=text_width, h=2 * text_width)
# save the figure in folder figures f folder does not exist create it

    output_dir = Path(__file__).resolve().parents[1] / "Report" / "Figures"
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # Create the directory if it does not exist
    output_dir_latex = Path(__file__).resolve().parents[1] / "Report" / "Latex"
    if not os.path.exists(output_dir_latex):
        os.makedirs(output_dir_latex)
    
    fig.savefig(output_dir / f"pos_{meas.meas_name[mnr]}_objective_params.png", dpi=300)

    fig.savefig(output_dir_latex / f"pos_{meas.meas_name[mnr]}_objective_params.pgf")





# %% LaTeX table containing all the objective parameters

# Create a LaTeX table with the objective parameters
# for each measurement.

# Calculate the one-number objective parameters for each measurement according to the ISO 3382-1:2009 standard.



with open(os.path.join(output_dir_latex, "objective_params_table.tex"), "w") as f:
    f.write("\\definecolor{lightgray}{gray}{0.8}\n")
    f.write("\\begin{table*}[ht]\n")
    f.write("\\centering\n")
    f.write("\\rowcolors{2}{lightgray}{white}\n")
    f.write("\\renewcommand{\\arraystretch}{1.2}\n")
    f.write("\\begin{tabular}{|c|c|c|c|c|c|c|c|}\n")
    f.write("\\hline\n")
    f.write(
        "Label & $T_{20}$ [s] & EDT [s] & $C_{80}$ [dB] & $D_{50}$ [dB] & $T_s$ [s] & E [dB] & DRR [dB] \\\\\n")
    f.write("\\hline\n")

    for idx, meas in enumerate(measurements):
        # Calculate the one-number objective parameters (average of 500 - 1000 Hz octave bands)
        one_number_params_x = [param[2:4].mean()
                               for param in meas.subband_params[:5]]
    
        f.write(
            f"{meas.meas_name[idx]} & {one_number_params_x[0]:.2f} & {one_number_params_x[1]:.2f} & {one_number_params_x[2]:.2f} & {one_number_params_x[3]:.2f} & {one_number_params_x[4]:.2f} & {meas.broadband_params[5].mean():.2f} & {meas.broadband_params[6].mean():.2f} \\\\\n")
        f.write("\\hline\n")
        f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\caption{}\n")
    f.write("\\label{tab:objective_params}\n")
    f.write("\\end{table*}\n")

# %%
