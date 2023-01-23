import numpy as np
import pandas as pd
import random
import os
import csv


def get_daily_sports_timeseries(path, person, activities, durations):
    directory = path
    daily_sports_data = []
    for filename in os.listdir(directory):
        sub_directory = os.path.join(directory, filename)
        for file in os.listdir(sub_directory):
            if file == 'p{}'.format(person):
                activity_data = []
                for datum in os.listdir(os.path.join(sub_directory, file)):
                    # with open(sub_directory+'\\p{}\\'.format(person)+datum, mode='r') as csv_file:
                    data = np.genfromtxt(sub_directory + '\\p{}\\'.format(person) + datum, delimiter=',')
                    # for row in data:
                    activity_data.append(data)

                daily_sports_data.append(activity_data)

    daily_sports_data = np.asarray(daily_sports_data)

    time_series, labels = construct_daily_sports_activity(daily_sports_data, activities, durations)
    return time_series, labels


def construct_daily_sports_activity(daily_sports_data, activities, durations):
    time_series = []
    labels = []

    for activity in activities:
        for duration in durations:
            clips = np.random.randint(low=0, high=60, size=duration)
            for n in range(duration * 125):
                labels.append(activity)
            for clip in clips:
                time_series.append(daily_sports_data[activity][clip])

    time_series = [item for sublist in time_series for item in sublist]

    time_series = np.asarray(time_series)
    labels = np.asarray(labels)
    return time_series.T, labels


# The 19 activities are:
# sitting (0): index 0,
# standing (1),
# lying on back and on right side (2 and 3),
# ascending and descending stairs (4 and 5),
# standing in an elevator still (6)
# and moving around in an elevator (7),
# walking in a parking lot (8),
# walking on a treadmill with a speed of 4 km/h (in flat and 15 deg inclined positions) (9 and 10),
# running on a treadmill with a speed of 8 km/h (11),
# exercising on a stepper (12),
# exercising on a cross trainer (13),
# cycling on an exercise bike in horizontal and vertical positions (14 and 15),
# rowing (16),
# jumping (17),
# and playing basketball (18): index 18

# activities = [0, 2, 3, 1, 8, 4, 14, 15, 14, 15, 1, 8, 11, 1, 12, 18, 5]
# durations = [10, 5, 7, 3, 30, 8, 25, 25, 25, 20, 10, 30, 30, 4, 40, 80, 8]

# time_series, labels = get_daily_sports_timeseries('.\\Data\\Daily Sports Activities\\', 1, activities, durations)

# print(time_series.shape)
# print(labels.shape)


def construct_uci_har_timeseries(data_list, person, activities, durations):
    assert (len(activities) == len(durations))

    time_series = [[], [], [], [], [], []]
    labels = []
    for i, activity in enumerate(activities):

        selected_data = data_list[0][(np.where((data_list[0][:, -1] == person) & (data_list[0][:, -2] == activity)))]
        x, y = selected_data.shape
        selected_clips = random.sample(range(0, x), durations[i])

        for j, data in enumerate(data_list):
            for clip in selected_clips:
                selected_data = data[(np.where((data[:, -1] == person) & (data[:, -2] == activity)))][:, :-2]
                time_series[j].append(selected_data[clip])

        for r in range(durations[i] * 128):
            labels.append(activity)

    time_series = np.asarray(time_series)
    time_series = np.reshape(time_series, (6, -1))
    labels = np.asarray(labels)

    return time_series, labels


def get_uci_har_dataset(path, activities, durations, person):
    d1 = np.genfromtxt(path + "\\Inertial Signals\\body_acc_x_train.txt")
    d2 = np.genfromtxt(path + "\\Inertial Signals\\body_acc_y_train.txt")
    d3 = np.genfromtxt(path + "\\Inertial Signals\\body_acc_z_train.txt")
    d4 = np.genfromtxt(path + "\\Inertial Signals\\body_gyro_x_train.txt")
    d5 = np.genfromtxt(path + "\\Inertial Signals\\body_gyro_y_train.txt")
    d6 = np.genfromtxt(path + "\\Inertial Signals\\body_gyro_z_train.txt")

    activity_labels = np.genfromtxt(path + "\\y_train.txt", dtype=int)
    person_labels = np.genfromtxt(path + "\\subject_train.txt", dtype=int)

    data_list = [d1, d2, d3, d4, d5, d6]

    combined_labels = np.stack((activity_labels, person_labels), axis=-1, )
    for i, data in enumerate(data_list):
        data_list[i] = np.concatenate((data, combined_labels), axis=1)

    time_series, labels = construct_uci_har_time_series(data_list, person, activities, durations)

    return time_series, labels


# activities = [1, 2, 1, 3, 1, 5, 4, 5, 1, 2, 1, 5, 6]
# durations = [10, 5, 15, 5, 8, 15, 30, 8, 20, 15, 20, 10, 6]

# time_series, labels = get_uci_har_dataset(".\\Data\\UCI_HAR_Dataset\\train", activities, durations, 1)