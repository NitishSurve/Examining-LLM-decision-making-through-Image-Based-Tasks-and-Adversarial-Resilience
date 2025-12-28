import numpy as np


class DataProcess:

    def __init__(self):
        pass

    @staticmethod
    def train_test_between_subject(data, indx_data, train_blocks, values=None):
        if values is None:
            values = ['reward', 'action', 'state']

        train = {}
        sdata = indx_data[indx_data.train == "train"]
        ids = sdata['id'].unique().tolist()

        for s_id in ids:
            sub_data = data[data.id == s_id]
            train[s_id] = []

            for t in train_blocks:
                sub_data_t = sub_data[sub_data.block.isin([t])]
                dicts = {}
                for v in values:
                    if v in sub_data_t:
                        dicts[v] = sub_data_t[v].values[np.newaxis, :]
                dicts['id'] = s_id
                dicts['block'] = t
                train[s_id].append(dicts)

        return train

    @staticmethod
    def get_max_seq_len(train):
        max_len = -np.inf  # ✅ Fixed for NumPy 2.0
        max_fmri = -np.inf

        for s_data in train.values():
            for t_data in s_data:
                reward_seq_len = t_data['reward'].shape[1]
                if reward_seq_len > max_len:
                    max_len = reward_seq_len

                if 'fmri_timeseries' not in t_data or t_data['fmri_timeseries'] is None:
                    max_fmri = None
                else:
                    fmri_seq_len = t_data['fmri_timeseries'].shape[1]
                    if fmri_seq_len > max_fmri:
                        max_fmri = fmri_seq_len

        return max_len, max_fmri

    @staticmethod
    def merge_blocks(data):
        merged_data = {}
        for k in sorted(data.keys()):
            v = data[k]
            merged_data[k] = DataProcess.merge_data({'merged': v})['merged']
        return merged_data

    @staticmethod
    def merge_data(train, batch_size=-1, vals=None):
        if vals is None:
            vals = ['action', 'reward', 'state']

        max_len, _ = DataProcess.get_max_seq_len(train)

        def app_not_None(arr, to_append, max_len):
            if to_append is not None:
                if max_len is not None:
                    pad_width = [(0, 0), (0, max_len - to_append.shape[1])] + [(0, 0)] * (to_append.ndim - 2)
                    arr.append(np.pad(to_append, pad_width, mode='constant', constant_values=(0, -1)))
                else:
                    arr.append(to_append)

        def none_if_empty(arr):
            return np.concatenate(arr) if arr else None

        dicts = {v: [] for v in vals}
        ids = []
        seq_lengths = []
        batches = []
        cur_size = 0

        for k_data in reversed(sorted(train.keys())):
            s_data = train[k_data]
            for t_data in s_data:
                # Get the shape length from the first value key (assuming it always exists)
                sample_key = next((key for key in vals if key in t_data), None)
                if sample_key is None:
                    continue
                seq_lengths.append(t_data[sample_key].shape[1])

                for v in vals:
                    app_not_None(dicts[v], t_data.get(v), max_len)
                ids.append(t_data['id'])

                cur_size += 1
                if batch_size != -1 and cur_size >= batch_size:
                    out_dict = {v: none_if_empty(dicts[v]) for v in vals}
                    out_dict['block'] = len(batches)
                    out_dict['id'] = ids
                    batches.append(out_dict)

                    dicts = {v: [] for v in vals}
                    ids = []
                    cur_size = 0

        if cur_size > 0:
            out_dict = {v: none_if_empty(dicts[v]) for v in vals}
            out_dict['block'] = len(batches)
            out_dict['id'] = ids
            out_dict['seq_lengths'] = np.array(seq_lengths)
            batches.append(out_dict)

        return {'merged': batches}
