# reshape class

import numpy as np

print('reshaping data...')

def reshape_CNN_data(data):
    r = 0
    total_rows = data.shape[0]
    list_df_train4D = []

    for i in range(1, total_rows, 720):
        end_loop = min(i + 719 - 60, total_rows - 59)

        for j in range(i, end_loop):
            # Inside the reshape_CNN_data function
            df_60s = np.flipud(data[j:j + 60, :])
            # Add the channel dimension
            df_60s = df_60s[np.newaxis, :, :, np.newaxis]
            # Reorder the dimensions to [batch_size, channels, height, width]
            df_60s = np.transpose(df_60s, (0, 3, 1, 2))

            list_df_train4D.append(df_60s)
            r += 1

    list_df_train4D = np.concatenate(list_df_train4D, axis=0)

    return list_df_train4D


def reshape_target_data(data):
    reshaped_data = []

    for i in range(0, data.shape[0], 660):
        upper_limit = min(i + 659, data.shape[0] - 1)

        for j in range(i, upper_limit + 1):
            reshaped_data.append(data[j])

    return np.array(reshaped_data)

def reshape_MLP_data(data, cnn_data):
    reshaped_data_list = []
    r = 0
    for i in range(0, data.shape[0], 660):
        upper_limit = min(i + 659, data.shape[0] - 1)
        for j in range(i, upper_limit + 1):
            if r >= len(cnn_data):
                break
            dft_60s = data[j][np.newaxis, :]  # Make dft_60s 2D with shape (1, num_features)
            ctdata = cnn_data[r]

            ct = ctdata[:, 0, :].flatten()  # Flatten to 1D

            ct = ct[np.newaxis, :]  # Make it 2D with shape (1, 60*some_value)

            dft_60s = np.concatenate((dft_60s, ct), axis=1)  # Concatenate along the second dimension

            reshaped_data_list.append(dft_60s)
            r += 1

    reshaped_data_array = np.array(reshaped_data_list)
    return reshaped_data_array
