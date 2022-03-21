# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd
from obspy import read
import os
import matplotlib.pyplot as plt
import plotly.express as px
from geopy import distance
from scipy.fft import fft, ifft, fftfreq

# # Data Cleaning

# +
directory = "./sw4_test_output/"
obs_directory = "./obs_test_output/"

# station location data
locations = pd.read_csv('sw4_station.txt', delimiter = ",")


# -

def create_df(directory):
    """
    Creates pandas DataFrame from seismic data.
    
    Args:
        directory: The directory (string) where seismic data is located (containing .e, ,n. .u files)
    
    Returns:
        pandas DataFrame
    
    Raises:
        AssertionError: If there is an issue with the data files
    """
    # cleaning and putting all the sw4 seismic data into a DataFrame
    # .BH is removed (they are duplicate sensors)
    file_list = os.listdir(directory)
    file_list.sort()
    file_list = [x for x in file_list if "sw4img" not in x and ".dat" not in x and "BH" not in x and ".xml" not in x and ".sacpz" not in x]
    assert((len(file_list) / 3) - int(len(file_list) / 3) == 0)
    df = pd.DataFrame(data=file_list)
    df.columns = ['Point']
    
    # peak and duration data for each point
    absolute_peaks = []
    durations = []
    amplitudes = []
    times = []
    for i in range(len(file_list)):
        st = read(directory+file_list[i])
        tr = st[0]
        abs_data = abs(tr.data)
        time = tr.times()
        threshold = max(abs_data) * 0.1
        start = np.argmax(abs_data>threshold)
        data_flipped = np.flip(abs_data)
        end = len(abs_data) - np.argmax(data_flipped>threshold) - 1
        duration = time[end-1] - time[start]
        absolute_peaks.append(abs(max(tr.data, key=abs)))
        durations.append(duration)
        amplitudes.append(tr.data)
        times.append(tr.times())

        # test single point
        # set name of data point to name
        """
        name = "BK.BRK.HN.u"
        if file_list[i] == name:
            print('peak: ', abs(max(tr.data, key=abs)))
            print('duration: ', [time[start], time[end]])
            plt.plot(time, tr.data)
            plt.axvline(x=time[start], color='r')
            plt.axvline(x=time[end], color='r')   
        """
    
    # full dataframe
    df["Station"] = [x[:-5] for x in df["Point"]]
    df["Amplitudes"] = amplitudes
    df["Times"] = times
    df["Absolute Peak"] = absolute_peaks
    df["Duration"] = durations
    df_with_locations = pd.merge(df, locations, left_on='Station', right_on='net.sta')
    df_with_locations = df_with_locations.drop(columns=['net.sta'])
    source = (37.86119, -122.24233)
    df_with_locations['Distance (miles)'] = df_with_locations.apply(lambda row: distance.distance((row.lat, row.lon), source).miles, axis=1)
    return df_with_locations


sw4_df = create_df(directory)
sw4_df

obs_df = create_df(obs_directory)
obs_df

# # Visualizing Peaks and Duration for Each Direction

# only .e
east_west = sw4_df.iloc[::3, :]
obs_east_west = obs_df.iloc[::3, :]
# only .n
north_south = sw4_df.iloc[1::3, :]
obs_north_south = obs_df.iloc[::3, :]
# only .u
up_down = sw4_df.iloc[1::3, :]
obs_up_down = obs_df.iloc[::3, :]

# +
ew = px.scatter_mapbox(east_west, lat='lat', lon='lon', color='Absolute Peak', hover_name='Station', mapbox_style='stamen-terrain', color_continuous_scale = 
                'rainbow', range_color=(0,0.003), title="{SW4} Peak Amplitude at Each Individual Station (East-West)")
ew.update_traces(marker={'size': 8})
ew.add_scattermapbox(lat=[37.86119], lon=[-122.24233], 
                     hovertemplate = 'SOURCE',
                     marker_size = 15,
                     marker_color = 'pink',
                     showlegend = False
                     )
ew.show()

obs_ew = px.scatter_mapbox(obs_east_west, lat='lat', lon='lon', color='Absolute Peak', hover_name='Station', mapbox_style='stamen-terrain', color_continuous_scale = 
                'rainbow', range_color=(0,0.003), title="{OBSERVED} Peak Amplitude at Each Individual Station (East-West)")
obs_ew.update_traces(marker={'size': 8})
obs_ew.add_scattermapbox(lat=[37.86119], lon=[-122.24233], 
                     hovertemplate = 'SOURCE',
                     marker_size = 15,
                     marker_color = 'pink',
                     showlegend = False
                     )
obs_ew.show()

# +
ns = px.scatter_mapbox(north_south, lat='lat', lon='lon', color='Absolute Peak', hover_name='Station', mapbox_style='stamen-terrain', color_continuous_scale = 
                'rainbow', range_color=(0,0.003), title="{SW4} Peak Amplitude at Each Individual Station (North-South)")
ns.update_traces(marker={'size': 8})

ns.show()

obs_ns = px.scatter_mapbox(obs_north_south, lat='lat', lon='lon', color='Absolute Peak', hover_name='Station', mapbox_style='stamen-terrain', color_continuous_scale = 
                'rainbow', range_color=(0,0.003), title="{OBSERVED} Peak Amplitude at Each Individual Station (North-South)")
obs_ns.update_traces(marker={'size': 8})

obs_ns.show()

# +
ud = px.scatter_mapbox(up_down, lat='lat', lon='lon', color='Absolute Peak', hover_name='Station', mapbox_style='stamen-terrain', color_continuous_scale = 
                'rainbow', range_color=(0, 0.003), title="{SW4} Peak Amplitude at Each Individual Station (Up-Down)")
ud.update_traces(marker={'size': 8})

ud.show()

obs_ud = px.scatter_mapbox(obs_up_down, lat='lat', lon='lon', color='Absolute Peak', hover_name='Station', mapbox_style='stamen-terrain', color_continuous_scale = 
                'rainbow', range_color=(0, 0.003), title="{OBSERVED} Peak Amplitude at Each Individual Station (Up-Down)")
obs_ud.update_traces(marker={'size': 8})

obs_ud.show()
# -

# There does not seem to be major differences between each direction within each dataset, although the east-west peaks seem to be on average higher than the other directions in the San Franciso area for the sw4 data. I believe we can safely take the magnitude of all 3 points and use 3D-vector data to do our analysis.
#
# However, when you compare the sw4 data with the observed data, you can clearly see that the observed data has significantly higher amplitudes than what is generated by the model.

# +
ew = px.scatter_mapbox(east_west, lat='lat', lon='lon', color='Duration', hover_name='Station', mapbox_style='stamen-terrain', color_continuous_scale = 
                'rainbow', range_color=(0,200), title="{SW4} Duration at Each Individual Station (East-West)")
ew.update_traces(marker={'size': 8})
ew.add_scattermapbox(lat=[37.86119], lon=[-122.24233], 
                     hovertemplate = 'SOURCE',
                     marker_size = 15,
                     marker_color = 'pink',
                     showlegend = False
                     )
ew.show()

obs_ew = px.scatter_mapbox(obs_east_west, lat='lat', lon='lon', color='Duration', hover_name='Station', mapbox_style='stamen-terrain', color_continuous_scale = 
                'rainbow', range_color=(0,200), title="{OBSERVED} Duration at Each Individual Station (East-West)")
obs_ew.update_traces(marker={'size': 8})
obs_ew.add_scattermapbox(lat=[37.86119], lon=[-122.24233], 
                     hovertemplate = 'SOURCE',
                     marker_size = 15,
                     marker_color = 'pink',
                     showlegend = False
                     )
obs_ew.show()

# +
ns = px.scatter_mapbox(north_south, lat='lat', lon='lon', color='Duration', hover_name='Station', mapbox_style='stamen-terrain', color_continuous_scale = 
                'rainbow', range_color=(0,200), title="{SW4} Duration at Each Individual Station (North-South)")
ns.update_traces(marker={'size': 8})
ns.add_scattermapbox(lat=[37.86119], lon=[-122.24233], 
                     hovertemplate = 'SOURCE',
                     marker_size = 15,
                     marker_color = 'pink',
                     showlegend = False
                     )
ns.show()

obs_ns = px.scatter_mapbox(obs_north_south, lat='lat', lon='lon', color='Duration', hover_name='Station', mapbox_style='stamen-terrain', color_continuous_scale = 
                'rainbow', range_color=(0,200), title="{OBSERVED} Duration at Each Individual Station (North-South)")
obs_ns.update_traces(marker={'size': 8})
obs_ns.add_scattermapbox(lat=[37.86119], lon=[-122.24233], 
                     hovertemplate = 'SOURCE',
                     marker_size = 15,
                     marker_color = 'pink',
                     showlegend = False
                     )
obs_ns.show()

# +
ud = px.scatter_mapbox(up_down, lat='lat', lon='lon', color='Duration', hover_name='Station', mapbox_style='stamen-terrain', color_continuous_scale = 
                'rainbow', range_color=(0,200), title="{SW4} Duration at Each Individual Station (Up-Down)")
ud.update_traces(marker={'size': 8})
ud.add_scattermapbox(lat=[37.86119], lon=[-122.24233], 
                     hovertemplate = 'SOURCE',
                     marker_size = 15,
                     marker_color = 'pink',
                     showlegend = False
                     )
ud.show()

obs_ud = px.scatter_mapbox(obs_up_down, lat='lat', lon='lon', color='Duration', hover_name='Station', mapbox_style='stamen-terrain', color_continuous_scale = 
                'rainbow', range_color=(0,200), title="{OBSERVED} Duration at Each Individual Station (Up-Down)")
obs_ud.update_traces(marker={'size': 8})
obs_ud.add_scattermapbox(lat=[37.86119], lon=[-122.24233], 
                     hovertemplate = 'SOURCE',
                     marker_size = 15,
                     marker_color = 'pink',
                     showlegend = False
                     )
obs_ud.show()


# -

# As you would probably expect after seeing the previous amplitude plots, the durations generated by sw4 are also much shorter than the observed data.

# # Visualizing as 3D Vector Data

def create_df_3d(directory, bandpass=False, bp_range=[0, 1]):
    """
    Creates pandas DataFrame from seismic data using total magnitude of all directions.
    
    Args:
        directory: The directory (string) where seismic data is located (containing .e, ,n. .u files)
        bandpass: Boolean denoting whether or not there is a band-pass filter applied.
        bp_range: List for the range of frequencies (between 0 and 1) for the band pass filter 
    
    Returns:
        pandas DataFrame
    
    Raises:
        AssertionError: If there is an issue with the data files   
    """
    file_list = os.listdir(directory)
    file_list.sort()
    file_list = [x for x in file_list if "sw4img" not in x and ".dat" not in x and "BH" not in x and ".xml" not in x and ".sacpz" not in x]
    assert((len(file_list) / 3) - int(len(file_list) / 3) == 0)
    
    # creating 3D Vector Data
    magnitudes = []
    durations = []
    peaks = []
    for i in range(0, len(file_list), 3):
        st = read(directory+file_list[i])
        tr_e = st[0]
        if bandpass:
            tr_e.filter('bandpass', freqmin=bp_range[0], freqmax=bp_range[1], corners=2, zerophase=True)
        x = tr_e.data
        st = read(directory+file_list[i+1])
        tr_n = st[0]
        if bandpass:
            tr_n.filter('bandpass', freqmin=bp_range[0], freqmax=bp_range[1], corners=2, zerophase=True)
        y = tr_n.data
        st = read(directory+file_list[i+2])
        tr_u = st[0]
        if bandpass:
            tr_u.filter('bandpass', freqmin=bp_range[0], freqmax=bp_range[1], corners=2, zerophase=True)
        z = tr_u.data
        length = min(len(x), len(y), len(z))
        mag = np.sqrt(x[:length]**2 + y[:length]**2 + z[:length]**2)
        threshold = max(mag) * 0.1
        start = np.argmax(mag>threshold)
        mag_flipped = np.flip(mag)
        end = len(mag) - np.argmax(mag_flipped>threshold) - 1
        time = tr_u.times()
        peak = max(mag)
        duration = time[end] - time[start]
        durations.append(duration)
        magnitudes.append(mag)
        peaks.append(peak)

        # test single point
        # set name of data point to name
        """
        name = "BK.BKS.HN"
        if file_list[i][:-2] == name:
            print(file_list[i][:-2])
            plt.plot(time, mag)
            plt.axvline(x=time[start], color='r')
            plt.axvline(x=time[end], color='r')
            print('duration: ', [time[start], time[end]])
        """
    
    # new DataFrame for 3D Data
    new_points = file_list[::3]
    new_points = [x[:-2] for x in new_points]
    df_3d = pd.DataFrame(data=new_points)
    df_3d.columns = ['Point']
    df_3d["Station"] = [x[:-3] for x in df_3d["Point"]]
    df_3d["Magnitudes"] = magnitudes
    df_3d["Peak"] = peaks
    df_3d["Duration"] = durations
    df_3d = pd.merge(df_3d, locations, left_on='Station', right_on='net.sta')
    df_3d = df_3d.drop(columns=['net.sta'])
    df_3d = df_3d.drop_duplicates(subset = ["Station"])
    return df_3d


sw4_df_3d = create_df_3d(directory)
sw4_df_3d

obs_df_3d = create_df_3d(obs_directory)
obs_df_3d

# +
fig1 = px.scatter_mapbox(sw4_df_3d, lat='lat', lon='lon', color='Peak', hover_name='Station', 
                         mapbox_style='stamen-terrain', color_continuous_scale = 
                        'rainbow', range_color=(0, 0.003),
                         title="{SW4} Peak Amplitude (Using 3D Vector Data) at Each Individual Station")
fig1.update_traces(marker={'size': 8})
fig1.add_scattermapbox(lat=[37.86119], lon=[-122.24233], 
                     hovertemplate = 'SOURCE',
                     marker_size = 15,
                     marker_color = 'pink',
                     showlegend = False
                     )
fig1.show()

obs_fig1 = px.scatter_mapbox(obs_df_3d, lat='lat', lon='lon', color='Peak', hover_name='Station', 
                         mapbox_style='stamen-terrain', color_continuous_scale = 
                        'rainbow', range_color=(0, 0.003),
                         title="{OBSERVED} Peak Amplitude (Using 3D Vector Data) at Each Individual Station")
obs_fig1.update_traces(marker={'size': 8})
obs_fig1.add_scattermapbox(lat=[37.86119], lon=[-122.24233], 
                     hovertemplate = 'SOURCE',
                     marker_size = 15,
                     marker_color = 'pink',
                     showlegend = False
                     )
obs_fig1.show()

# +
fig2 = px.scatter_mapbox(sw4_df_3d, lat='lat', lon='lon', color='Duration', mapbox_style='stamen-terrain', hover_name='Station', color_continuous_scale = 
                'rainbow', range_color=(0, 200), title="{SW4} Duration of Earthquake (Using 3D Vector Data) at Each Individual Station")
fig2.update_traces(marker={'size': 8})
fig2.add_scattermapbox(lat=[37.86119], lon=[-122.24233], 
                     hovertemplate = 'SOURCE',
                     marker_size = 15,
                     marker_color = 'pink',
                     showlegend = False
                     )
fig2.show()

obs_fig2 = px.scatter_mapbox(obs_df_3d, lat='lat', lon='lon', color='Duration', mapbox_style='stamen-terrain', hover_name='Station', color_continuous_scale = 
                'rainbow', range_color=(0, 200), title="{OBSERVED} Duration of Earthquake (Using 3D Vector Data) at Each Individual Station")
obs_fig2.update_traces(marker={'size': 8})
obs_fig2.add_scattermapbox(lat=[37.86119], lon=[-122.24233], 
                     hovertemplate = 'SOURCE',
                     marker_size = 15,
                     marker_color = 'pink',
                     showlegend = False
                     )
obs_fig2.show()
# -

# Peak amplitude definitely shows a trend of decreasing as we move away from the source of the earthquake, duration seem to be determined by the terrain.
#
# Again though, the observed data is pretty drastically different from the sw4 output.

# # Plotting Peak Amplitude/Duration Vs Distance

source = (37.86119, -122.24233)
sw4_df_3d['Distance (miles)'] = sw4_df_3d.apply(lambda row: distance.distance((row.lat, row.lon), source).miles, axis=1)
fig = px.scatter(sw4_df_3d, x = 'Distance (miles)', y = 'Peak', title = "{SW4} Peak Magnitude (3D Vector) vs Distance");
fig.show()

fig = px.scatter(sw4_df_3d, x ='Distance (miles)', y ='Duration', title = "{SW4} Duration vs Distance", color_discrete_sequence=['red']);
fig.show()

obs_df_3d['Distance (miles)'] = obs_df_3d.apply(
    lambda row: distance.distance((row.lat, row.lon), source).miles, axis=1)
fig = px.scatter(obs_df_3d, x = 'Distance (miles)', y = 'Peak', title = "{OBSERVED} Peak Magnitude (3D Vector) vs Distance");
fig.update_yaxes(range=[0, 0.04])
fig.show()
# note - there is one very far outlier at about ~65 miles if you rescale

fig = px.scatter(obs_df_3d, x = 'Distance (miles)', y = 'Duration', title = "{OBSERVED} Duration vs Distance", color_discrete_sequence=['red']);
fig.update_yaxes(range=[0, 120])
fig.show()


# Definitely a more interesting trend going on for the observed data here. Distance seems to not correlate with the duration, but there are certain areas where the duration is almost exactly the same (possibly along a line on the map). Or our method of computing duration just does not work for the observed data.

# # Comparing Data Points

# Here we compare NC.C033 and BK.BRK, because from the peak amplitude geographic plot it seems like BK.BRK is a very small value, even though it is about the same distance (very close) to the source. There could be an error in our computation.

def compare(point1, point2):
    """
    Plots out graphs for two different points you want to compare.
    
    Args:
    point1: [name, table]; name=Name of data point, table=pandas DataFrame where data point comes from
    point2: [name, table]
    
    Returns:
        None, just plots
    """
    df1 = point1[1]
    df1 = df1.set_index('Point')
    point1 = point1[0]
    df2 = point2[1]
    df2 = df2.set_index('Point')
    point2 = point2[0]
    
    T = 0.037679 # delta seems to be the same for all points

    data1 = df1.loc[point1]['Amplitudes']
    times1 = df1.loc[point1]['Times']
    data2 = df2.loc[point2]['Amplitudes']
    times2 = df2.loc[point2]['Times']
    
    plt.figure(0)
    plt.plot(times1, data1)
    plt.ylabel("Amplitude")
    plt.xlabel("Time")
    plt.title(point1);

    plt.figure(1)
    plt.plot(times2, data2, color='orange')
    plt.ylabel("Amplitude")
    plt.xlabel("Time")
    plt.title(point2);
    
    # For Fourier Transform, can do other visualizations
    N = len(data1)
    yf = fft(data1)
    xf = fftfreq(N, T)[:N//2]
    yf = 2.0/N * np.abs(yf[0:N//2])
    plt.figure(2)
    plt.plot(xf, yf, label='point1')
    N = len(data2)
    yf = fft(data2)
    xf = fftfreq(N, T)[:N//2]
    yf = 2.0/N * np.abs(yf[0:N//2])
    plt.figure(2)
    plt.plot(xf, yf, label='point2')
    plt.grid()
    plt.ylabel("Amplitude")
    plt.xlabel("Frequency (Hz)");
    plt.xlim([0,2.5])
    plt.title(point1 + ', ' + point2);
    plt.legend()


compare(['BK.BRK.HN.u', sw4_df], ['NC.C033.HN.u', sw4_df])

# It seems like NC.C033 really does have a higher peak (it is very extreme at the start). BK.BRK actually has relatively low amplitudes despite being so close to the source, which is interesting. We will know if the model is accurate in simulating this once we have the true data.

compare(['NC.JBR.HN.e', sw4_df], ['NC.JBR.HN.e', obs_df])

# Way different even though it is the same point.

# # Frequency Analysis

# +
indexed = sw4_df.set_index('Point')
frequency_table = []
Corr_Amp = []
for i in sw4_df.Point:  
    data = indexed.loc[i]['Amplitudes']
    N = len(data)
    yf = fft(data)
    xf = fftfreq(N, T)[:N//2]
    yf = 2.0/N * np.abs(yf[0:N//2])
    peak_amp_at = np.argmax(yf)
    frequency = xf[peak_amp_at]
    amp = yf[peak_amp_at]
    point = (frequency, amp)

    peak_amp_at2 = np.argsort(yf, axis=0)[-2]
    frequency2 = xf[peak_amp_at2]
    amp2 = yf[peak_amp_at2]
    point2 = (frequency2, amp2)

    highest_points = pd.DataFrame([point, point2])
    firstfreq = highest_points.min()[0]
    frequency_table.append(firstfreq)
    amplification = highest_points.min()[1]
    Corr_Amp.append(amplification)
    
    
sw4_df["Frequency"] = frequency_table 
sw4_df["Corr_Amp"] = Corr_Amp

sw4_df_u = sw4_df[sw4_df.Point.str.contains('.u')]
sw4_df_n = sw4_df[sw4_df.Point.str.contains('.n')]
sw4_df_e = sw4_df[sw4_df.Point.str.contains('.e')]
# -

sw4_df_e.plot(x = 'Distance (miles)', y = 'Frequency', kind = 'scatter', ylabel = 'Frequency (Hz)', xlabel = 'Distance (miles)', title = "e - Frequency of Peak Amplitude vs Distance")
fig_e = px.scatter_mapbox(sw4_df_e, lat='lat', lon='lon', color='Frequency', hover_name='Station', 
                         mapbox_style='stamen-terrain', color_continuous_scale = 
                        'rainbow', 
                         title="Frequency Value of Peak Amplitude at Each Individual Station (East-West)")
fig_e.update_traces(marker={'size': 8})
fig_e.add_scattermapbox(lat=[37.86119], lon=[-122.24233], 
                     hovertemplate = 'SOURCE',
                     marker_size = 15,
                     marker_color = 'pink',
                     showlegend = False
                     )
fig_e.show();

sw4_df_n.plot(x = 'Distance (miles)', y = 'Frequency', kind = 'scatter', ylabel = 'Frequency (Hz)', xlabel = 'Distance (miles)', title = "n - Frequency of Peak Amplitude vs Distance", color="r")
fig_n = px.scatter_mapbox(sw4_df_n, lat='lat', lon='lon', color='Frequency', hover_name='Station', 
                         mapbox_style='stamen-terrain', color_continuous_scale = 
                        'rainbow', 
                         title="Frequency Value of Peak Amplitude at Each Individual Station (North-South)")
fig_n.update_traces(marker={'size': 8})
fig_n.add_scattermapbox(lat=[37.86119], lon=[-122.24233], 
                     hovertemplate = 'SOURCE',
                     marker_size = 15,
                     marker_color = 'pink',
                     showlegend = False
                     )
fig_n.show()

sw4_df_u.plot(x = 'Distance (miles)', y = 'Frequency', kind = 'scatter', ylabel = 'Frequency (Hz)', xlabel = 'Distance (miles)', title = "u - Frequency of Peak Amplitude vs Distance", color="g")
fig_u = px.scatter_mapbox(sw4_df_u, lat='lat', lon='lon', color='Frequency', hover_name='Station', 
                         mapbox_style='stamen-terrain', color_continuous_scale = 
                        'rainbow', 
                         title="Frequency Value of Peak Amplitude at Each Individual Station (Up-Down)")
fig_u.update_traces(marker={'size': 8})
fig_u.add_scattermapbox(lat=[37.86119], lon=[-122.24233], 
                     hovertemplate = 'SOURCE',
                     marker_size = 15,
                     marker_color = 'pink',
                     showlegend = False
                     )
fig_u.show()

df_3d_bp = create_df_3d(directory, True, [0.2, 0.5])

df_3d_bp1 = create_df_3d(directory, True, [0.5, 1])

fig1 = px.scatter_mapbox(df_3d_bp, lat='lat', lon='lon', color='Peak', hover_name='Station', 
                         mapbox_style='stamen-terrain', color_continuous_scale = 
                        'rainbow', range_color=(0, 0.002),
                         title="{SW4} Peak Amplitude (Using Band-Passed [0.2-0.5] 3D Vector Data) at Each Individual Station")
fig1.update_traces(marker={'size': 8})
fig1.add_scattermapbox(lat=[37.86119], lon=[-122.24233], 
                     hovertemplate = 'SOURCE',
                     marker_size = 15,
                     marker_color = 'pink',
                     showlegend = False
                     )
fig1.show()

fig2 = px.scatter_mapbox(df_3d_bp1, lat='lat', lon='lon', color='Peak', hover_name='Station', 
                         mapbox_style='stamen-terrain', color_continuous_scale = 
                        'rainbow', range_color=(0, 0.002),
                         title="{SW4} Peak Amplitude (Using Band-Passed [0.5-1.0] 3D Vector Data) at Each Individual Station")
fig2.update_traces(marker={'size': 8})
fig2.add_scattermapbox(lat=[37.86119], lon=[-122.24233], 
                     hovertemplate = 'SOURCE',
                     marker_size = 15,
                     marker_color = 'pink',
                     showlegend = False
                     )
fig2.show()

create_df_3d(obs_directory, True, [0.2, 0.5])
# synthetic/ observation
