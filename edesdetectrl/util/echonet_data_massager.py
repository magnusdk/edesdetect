import math
import pandas as pd


def calculate_tracing_volumes(volumetracings, filename, filter_tracings_count=21):
    """
    Take volumetracings dataframe and a filename (excluding .avi extension)
    and return a dictionary from frame to pseudo-volume.

    Pseudo-volume means that it is only a proxy for comparing two volumes,
    but that it is not the actual volume.
    """
    filename = filename + ".avi"
    volumetracing = volumetracings[volumetracings["FileName"] == filename]

    frame_volumes = {}
    volumetracing_grouped = volumetracing.groupby(["Frame"])

    if not (volumetracing_grouped.count() == filter_tracings_count).all().all():
        return None
    if not volumetracing_grouped.ngroups == 2:
        return None

    for frame, vt_data in volumetracing_grouped:
        volume = 0  # Volume is calculated as the sum of the lengths of all line tracings for a given frame.
        for _, row in vt_data.iterrows():
            volume += math.sqrt(
                (row["X2"] - row["X1"]) ** 2 + (row["Y2"] - row["Y1"]) ** 2
            )
        frame_volumes[frame] = volume

    return frame_volumes


def get_ed_es_from_volumes(volumes_dict):
    """
    Take a dictionary of two frames and their volume and return a dictionary
    of 'ED' and 'ES' frames.

    This function assumes that there are only two frames present in volumes_dict.
    """
    ed_frame = max(volumes_dict, key=lambda i: volumes_dict[i])
    es_frame = min(volumes_dict, key=lambda i: volumes_dict[i])
    return {"ED": ed_frame, "ES": es_frame}


def convert_to_better_format(
    filelist_csv_file, volumetracings_csv_file, output_filename
):
    filelist_df = pd.read_csv(filelist_csv_file)
    volumetracings_df = pd.read_csv(volumetracings_csv_file)
    with open(output_filename, "w") as f:
        f.write("FileName,ED_Frame,ES_Frame\n")
        for output_filename in filelist_df["FileName"].values:
            volumes = calculate_tracing_volumes(volumetracings_df, output_filename)
            if volumes is None:
                continue
            ed_es = get_ed_es_from_volumes(volumes)
            f.write(f'{output_filename},{ed_es["ED"]},{ed_es["ES"]}\n')


def main_convert_to_better_format(
    filelist_path="/home/magnus/research/data/EchoNet-Dynamic/FileList.csv",
    volumetracings_path="/home/magnus/research/data/EchoNet-Dynamic/VolumeTracings.csv",
    output_path="processed_labels.csv",
):
    convert_to_better_format(filelist_path, volumetracings_path, output_path)
