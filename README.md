# Camera-Based Heart Rate and Heart Rate Variability Measurement (CBHRM)

## Overview
This project is part of a Bachelor's thesis focused on real-time heart rate and variability detection using camera-based methods. It aims to provide a non-invasive way to measure heart rate and heart rate variability (HRV) using image processing and signal analysis techniques. Application based on proposed pipeline[1]

> __Important__: OpenCV needs to be build from source with Gstreamer

## Features
- **Heart Rate Monitoring**: Utilizes camera input to detect and monitor heart rate.
- **Heart Rate Variability Analysis**: Offers HRV analysis using advanced signal processing techniques.
- **Non-Invasive Method**: Employs a camera-based approach, eliminating the need for physical contact.
- **Configurable Settings**: Includes various settings for camera resolution, filtering parameters, and more, as defined in `settings.toml`.
- **Interactable Dashboard**: - Dashboard for recording and visualising various signals inclusing: 
    - HR (with or without reference device)
    - HRV (with or without reference device)
    - rPPG Signal 
    - Post Processed rPPG signal
    - Correlation Plots 
    - ...

- **Reference Device evaluation**: utilises [Openant](https://github.com/Tigge/openant)
 to read in ANT+ compatible chest strap
- **Evaluation of Accuracy**: Evaluates the reference device HR against the estimated HR
- **Reference Device HRV calulation**: Caluculates the HRV of a ANT+ compatible chest strap
## Installation
1. Clone the repository:

    ```shell
    git clone https://github.com/parisj/CBHRM.git
    ```
2. Install required dependencies:
    ```shell
    pip install -r requirements.txt
    ```
## Usage
To run the application, execute:
```shell
cd CBHRM
python run_application.py

```



## Configuration Settings

### Camera Settings
- `fps_camera`: The frame rate of the camera in frames per second. Default is `20`.
- `resolution`: The resolution of the camera. Default is `960x720` pixels.
### Video Source Settings
- `live`: Determines if the video source is live (`true`) or pre-recorded (`false`). Default is `false`.
- `path`: The file path to the pre-recorded video. Default is `"dataset/P1/Talking/recording_3.avi"`.
- `evaluate_dataset`: A boolean flag to indicate whether to evaluate the dataset. Default is `true`.
- `path_measurements`: File path to the dataset measurements. Default is `"dataset/P1/Talking/recording_3.csv"`.
    #### For using the live webcam stream: 
    ```toml
    live = true
    evaluate_dataset = false
    ```
    #### for using prerecorded video: 
    ```toml
    live = false
    evaluate_dataset = false
    ```
    #### for evaluating a prerecorded dataset 
    ```toml
    live = false
    evaluate_dataset = true
    ```
### Filtering Parameters
- `fs`: Sampling frequency. Default is `20`.
- `lowpass_order`: Order of the lowpass filter. Default is `6`.
- `wn_lowpass`: Normalized cutoff frequency for the lowpass filter. Default is `0.6`.
- `pos_window_l`: Position window length. Default is `32`.
- `cutoff_window`: Cutoff window frequencies. Default is `[0.235, -0.235]`.
- `bandpass_order`: Order of the bandpass filter. Default is `25`.
### Heart Rate Measurement Settings
- `time_window`: Time window for measurement in milliseconds. Default is `200`.
- `start_delay_peak_detection`: Delay before starting peak detection in milliseconds. Default is `15`.
- `distance`: Minimum distance between peaks. Default is `6`.
- `prominence`: Minimum prominence of peaks. Default is `0`.
### Evaluation Parameters
- `delay`: Delay in evaluation in samples. Default is `85`.
- `len_hr_min`: Minimum length for heart rate data. Default is `21`.
- `fps`: Frames per second for evaluation. Default is `20`.
- `dashboard_refresh_time`: Dashboard refresh rate in milliseconds. Default is `1000`.
- `min_ref_measurements`: Minimum reference measurements. Default is `500`.
- `calibration_time_ref`: Calibration time for reference in samples. Default is `300`.
- `min_ref_hrv`: Minimum reference for HRV. Default is `90`.
- `calibration_time_ref_hrv`: Calibration time for HRV reference in samples. Default is `50`.
### Result Configuration
- `path`: Path to save results. Default is `"dataset/P2/breathing2.csv"`.
- `write`: Enable writing results. Default is `true`.
### Reference Measurement Settings
- `hrv_window`: HRV measurement window in samples. Default is `200`.
- `device_id`: Device ID for the reference measurement. Default is `20074`.
### Record Data Set Settings
- `fps_camera`: Frames per second for the camera in the dataset recording. Default is `20`.
- `resolution`: Camera resolution for the dataset recording. Default is `[960,720]`.
- `device_id`: Device ID for the dataset recording. Default is `20074`.
- `video_path`: Path to the video file for the dataset recording. Default is `"dataset/P2/breathing2.avi"`.
- `readings_path`: Path to the readings file for the dataset recording. Default is `"dataset/P2/breathing2.csv"`.
### Add new reference device
1. Scan with openant for devices and write json will found devices
```shell
openant scan --outfile devices.json
```
2. Extract device ID
3. Replace device ID with new ID

    > __Tip__: Duplicate settings.toml and name old one: settings_old_device_ID.toml
    

### Folder Strucutre
~~~
CBHRM/
    .gitignore
    README.md
    requirements.txt
    run_application.py
    settings.toml
    dataset/
        P1/
            Post_Workout/
            Rotation/
            Steady/
            Talking/
        P2/

            Rotation/
            Steady/
            Talking/
        P3/
            Post_Workout/
            Rotation/
            Steady/
            Talking/
        P4/
            result.csv
            Post_Workout/
            Rotation/
            Steady/
            Talking/
        Plots/
    results/
    src/
        blackboard.py
        control.py
        dashboard.py
        heart_rate_monitor.py
        image_processor.py
        signal_processor.py
        video_stream.py
        __init__.py
        devices/
            devices.json
        util/
            heart_rate.py
            histogram_inspection.py
            legacy_dashboard.py
            livefilter.py
            load_pkl.py
            plot_frequency.py
            plot_functions.py
            recoord_dataset.py
~~~



------------------------

 [1] A. Gudi, M. Bittner, R. Lochmans, and J. van Gemert, ‘Efficient Real-Time Camera Based Estimation of Heart Rate and Its Variability’. arXiv, Sep. 03, 2019. Accessed: Dec. 10, 2023. [Online]. Available: http://arxiv.org/abs/1909.01206
