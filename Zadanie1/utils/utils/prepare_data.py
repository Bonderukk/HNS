import pandas as pd
import glob
import numpy as np
from numpy.fft import fft
import json
import matplotlib.pyplot as plt 
from pathlib import Path

sensor = 'ACC'
terrain_labels ={
    1 : 'concrete',
    2 : 'grass',
    3 : 'pebbles',
    4 : 'sand',
    5 : 'paving stone',
    6 : 'synthetic running track',
}

def create_sliding_windows(data, window_size, step_size):
    windows = []
    labels = []
    
    for i in range(0, len(data) - window_size + 1, step_size):
        window = data[i:i+window_size]  # Extract a window of specified size
        windows.append(window)

    return np.array(windows)

def split_time_series(data, sampling_frequency=400):
    """
    Splits the input time series data into 1-second chunks.
    
    Parameters:
    - data (list): The time series data as a list of sensor measurements.
    - sampling_frequency (int): The sampling frequency of the sensor in Hz (default is 500 Hz).
    
    Returns:
    - list of lists: Each list contains 1 second of measurement data.
    """
    # Number of samples in 1 second
    samples_per_second = sampling_frequency
    
    # Calculate the number of full 1-second chunks
    num_full_chunks = len(data) // samples_per_second
    
    # Split the data into full 1-second chunks
    one_second_chunks = [data[i * samples_per_second: (i + 1) * samples_per_second] 
                         for i in range(num_full_chunks)]
    
    return one_second_chunks

def save_fft_to_json(fft_data,filename):
    json_data = []
    for terrain_type, signals in fft_data.items():

        for signal in signals:
            magnitudes = [abs(item) for item in signal]
            phases = [np.angle(item) for item in signal]

            json_data.append({
                'terrain_type' : terrain_type,
                'magnitudes' : magnitudes,
                'phases' : phases
            })

    with open(filename, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)
ROOT_DIR = r'C:\Codes\outdoor_terrain_classification_database\Measurements'


def save_fft_to_csv(data,sensor = 'ACC'):
    main_directory = Path(sensor)
    main_directory.mkdir(parents=True, exist_ok=True)

    for terrain_type, signals in data.items():
        counter = 0
        sub_dir = main_directory / f'{terrain_type}'
        sub_dir.mkdir(parents=True, exist_ok=True)
        for signal in signals:
            magnitudes = [abs(item) for item in signal]
            phases = [np.angle(item) for item in signal]

            sample_dict = {
                'magnitude': magnitudes,
                'phase' : phases,
            }
            csv_file = sub_dir / f'{counter}.csv'
            sample_df = pd.DataFrame(sample_dict)
            sample_df.to_csv(csv_file)
            counter += 1


if __name__ == '__main__':


    accelerometer_path_dict = dict()
    for item in terrain_labels.keys():
        acc_type = glob.glob(ROOT_DIR + f'\\*{sensor}_{item}_2*')
        accelerometer_path_dict[terrain_labels.get(item)] = acc_type

    accelerometer_data_dict = dict()
    for terrain_type,paths in accelerometer_path_dict.items():
        terrain_data = []
        for path in paths:
            df = pd.read_csv(path, header=None)
            z_axis = df.iloc[:,2].values
            z_axis_chunks = create_sliding_windows(data=z_axis, window_size=400,step_size=70)

            terrain_data = [*terrain_data, *z_axis_chunks]

        accelerometer_data_dict[terrain_type] = terrain_data

    accelerometer_fft_dict = dict()
    plt.figure(1)
    plt.plot(accelerometer_data_dict["concrete"][1])

    for terrain_type, signals in accelerometer_data_dict.items():
        fft_signals = [fft(f) for f in signals]

        accelerometer_fft_dict[terrain_type] = fft_signals

    plt.figure(2)
    plt.plot(abs(accelerometer_fft_dict['concrete'][1]))
    plt.show()


    #save_fft_to_json(accelerometer_fft_dict, 'accelerometer_fft_data.json')
    save_fft_to_csv(
        data = accelerometer_fft_dict,
        sensor= sensor
    )