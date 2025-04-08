import numpy as np

def merge_flashes(flashes,merge_threshold=1.0,time_method='min'):
    """Merge flashes from a list of flash events.

    Parameters
    ----------
    flashes : list
        List of flashes

    Returns
    -------
    merged_flash_event : FlashEvent
        Merged flash event
    """
    old_flashes = flashes.copy()
    # print('Old flashes:')
    # print('-'*50)
    # for f in old_flashes:
    #     print(f'id: {f.id}, time: {f.time}, volume_id: {f.volume_id}, total_pe: {f.total_pe}, center: {f.center}, width: {f.width}, time_width: {f.time_width}\n')
    flashes_to_pop = []
    for i,flash in enumerate(flashes):
        for j,other_flash in enumerate(flashes):
            if i != j and flash.volume_id != other_flash.volume_id and j not in flashes_to_pop and i not in flashes_to_pop:
                if np.abs(flash.time - other_flash.time) < merge_threshold:
                    flash.merge(other_flash, time_method=time_method)
                    #Remove the other flash
                    flashes_to_pop.append(j)
    #Remove the flashes that were merged
    for i in sorted(flashes_to_pop, reverse=True):
        flashes.pop(i)
    #Reassign the flash id
    for i,flash in enumerate(flashes):
        flash.id = i
        flash.volume_id = 0 #TODO: Set to whatever level of volume you're merging to. If we're merging at TPC level to the module lebel this should be the module id.
    #DEBUG - print the old and new flashes
    # print('New flashes:')
    # print('-'*50)
    # for f in flashes:
    #     print(f'id: {f.id}, time: {f.time}, volume_id: {f.volume_id}, total_pe: {f.total_pe}, center: {f.center}, width: {f.width}, time_width: {f.time_width}\n')
    # print('-'*50)
    # print(f'Deleted flashes: {flashes_to_pop}')
    # print('-'*50)
    return flashes
        

