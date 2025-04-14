import numpy as np
import copy

def merge_flashes(flashes, merge_threshold=1.0, time_method='min'):
    """Merge flashes from a list of flash events and track original IDs.

    Parameters
    ----------
    flashes : list
        List of flash objects (e.g., FlashEvent). Assumes each flash has
        an 'id' attribute and a 'merge' method.
    merge_threshold : float, optional
        Time difference threshold (in seconds or consistent units with flash.time)
        for merging flashes from different volumes, by default 1.0.
    time_method : str, optional
        Method used to determine the time of the merged flash ('min', 'max', 'weighted', etc.),
        passed to the flash.merge method, by default 'min'.

    Returns
    -------
    flashes : list
        List of merged flashes
    flash2oldflash_dict : dict
        Dictionary of flash to old flash mapping
    """
    if not flashes: # Handle empty input list
        return [], {}
    
    #Initialize the list of old flashes and the dictionary of flash to old flash mapping
    old_flashes = copy.deepcopy(flashes)
    unique_volume_ids = [0,1] #TODO: Use the detector geometry to get the unique volume ids
    flash2oldflash_dict = {f.id: [f]+[None]*(len(unique_volume_ids)-1) for f in old_flashes}

    flashes_to_pop = []
    for i,flash in enumerate(flashes):
        for j,other_flash in enumerate(flashes):
            if i != j and flash.volume_id != other_flash.volume_id and j not in flashes_to_pop and i not in flashes_to_pop:
                #Check if the two flashes are compatible in time, if so merge them
                if np.abs(flash.time - other_flash.time) < merge_threshold:
                    flash2oldflash_dict[old_flashes[i].id] = [old_flashes[i],old_flashes[j]]
                    flash.merge(other_flash, time_method=time_method)
                    #Remove the other flash
                    flashes_to_pop.append(j)

    #Remove the flashes that were merged
    for i in sorted(flashes_to_pop, reverse=True):
        flashes.pop(i)
        #Remove the flashes from the dictionary and reassign the ids
        flash2oldflash_dict.pop(i)
    #Reassign the keys for the dictionary
    flash2oldflash_dict = {i: flash2oldflash_dict[f.id] for i,f in enumerate(flashes)}
    #Reassign the flash id
    for i,flash in enumerate(flashes):
        flash.id = i
        flash.volume_id = 0 #TODO: Set to whatever level of volume you're merging to. If we're merging at TPC level to the module lebel this should be the module id.
    return flashes, flash2oldflash_dict