import mne
import os
import numpy as np
import sn_config as c


def main(subjects, data_path, task, stim_delay,
         category_code, event_id, t_min, t_max, reject):
    for i in np.arange(len(subjects)):
        print("participant: ", i)
        meg = subjects[i]
        filename = data_path + meg + \
            f"block_{task}_tsss_notch_BPF0.1_45_ICAeog_ecg_raw.fif"
        raw = mne.io.Raw(filename, preload=True)
        picks = mne.pick_types(raw.info, meg=True, eeg=True, eog=False,
                               stim=False)
        events = mne.find_events(raw, stim_channel="STI101",
                                 min_duration=0.001, shortest_event=1)
        # Considering the device(!) delay
        events[:, 0] += np.round(raw.info["sfreq"] * stim_delay)
        # Finding events with false responses
        for e in range(events.shape[0] - 2):
            if task == "LD":
                if events[e, 2] in category_code and events[e+2, 2] != 16384:
                    events[e, 2] = 7777
                elif events[e, 2] in np.array([6, 7, 9]) \
                        and events[e+2, 2] != 4096:
                    events[e, 2] = 8888
            else:
                if events[e, 2] in category_code and events[e+2, 2] > 100:
                    events[e, 2] = 7777
                elif events[e, 2] == 8 and events[e+2, 2] < 100:
                    events[e, 2] = 8888
        # Extracting epochs from a raw instance
        epochs = mne.Epochs(raw, events, event_id, t_min, t_max, picks=picks,
                            proj=True, baseline=(t_min, 0), reject=reject)
        # checking for the existence of desired directory to save the data
        if not os.path.isdir(data_path + meg):
            os.makedirs(data_path + meg)
        output = data_path + meg + f"block_{task}_epochs-epo.fif"
        # saving epochs
        epochs.save(output, overwrite=True)


if __name__ == "__main__":
    tasks = ["fruit", "odour", "milk", "LD"]

    for t in tasks:
        # Events info
        if t == "LD":
            event = c.event_id_ld
        else:
            event = c.event_id_sd

        main(c.subjects, c.data_path, t, c.stim_delay,
             c.category_code, event, c.tmin, c.tmax, c.reject)


subjects, data_path, task, stim_delay, category_code, event_id, t_min, t_max, reject = c.subjects, c.data_path, t, c.stim_delay, c.category_code, event, c.tmin, c.tmax, c.reject
