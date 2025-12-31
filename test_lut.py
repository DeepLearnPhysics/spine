import h5py
import pandas as pd

from lut import LUT


def plot_light_traps(data):
    """Plot optical detectors"""
    drawn_objects = []

    # print(data["/geometry_info/det_bounds/data"].shape)
    # print(data["/geometry_info/det_bounds/data"][:])

    print(data["/geometry_info/det_id/data"].shape)
    print(data["/geometry_info/det_id/data"][:])

    det_bounds = LUT.from_array(
        data["/geometry_info/det_bounds"].attrs["meta"],
        data["/geometry_info/det_bounds/data"],
    )
    sipm_rel_pos = LUT.from_array(
        data["geometry_info/sipm_rel_pos"].attrs["meta"],
        data["geometry_info/sipm_rel_pos/data"],
    )
    sipm_abs_pos = LUT.from_array(
        data["geometry_info/sipm_abs_pos"].attrs["meta"],
        data["geometry_info/sipm_abs_pos/data"],
    )
    det_ids = LUT.from_array(
        data["geometry_info/det_id"].attrs["meta"],
        data["geometry_info/det_id/data"],
    )

    sipm_keys = list(zip(sipm_rel_pos.keys()[0], sipm_rel_pos.keys()[1]))
    data = []
    for k in sipm_keys:
        sipm_x, sipm_y, sipm_z = sipm_abs_pos[k][0]
        tpc, side, sipm_pos = sipm_rel_pos[k][0]
        adc, channel = k
        det_id = det_ids[k][0]
        (min_x, min_y, min_z), (max_x, max_y, max_z) = det_bounds[(tpc, det_id)][0]
        row = {
            "det_id": det_id,
            "sipm_x": sipm_x,
            "sipm_y": sipm_y,
            "sipm_z": sipm_z,
            "tpc": tpc,
            "side": side,
            "sipm_pos": sipm_pos,
            "adc": adc,
            "channel": channel,
            "min_x": min_x,
            "min_y": min_y,
            "min_z": min_z,
            "max_x": max_x,
            "max_y": max_y,
            "max_z": max_z,
        }
        data.append(row)

    lut_based_channel_map = pd.DataFrame(data)
    print(lut_based_channel_map[:16])


with h5py.File("MiniRun6.5_1E19_RHC.flow.0000000.FLOW.hdf5") as data:
    det_ids = LUT.from_array(
        data["geometry_info/det_id"].attrs["meta"],
        data["geometry_info/det_id/data"],
    )

    sipm = LUT.from_array(
        data["geometry_info/sipm_rel_pos"].attrs["meta"],
        data["geometry_info/sipm_rel_pos/data"],
    )

    # print(data["geometry_info/det_id/data"].shape)
    # print(det_ids)
    # print(det_ids.keys())
    # print(sipm.keys())
    # print(det_ids[sipm.keys()])
    # print(det_ids._data)
    # print(det_ids._filled)
    # print(det_ids[0,10])

    plot_light_traps(data)
