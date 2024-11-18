from spine.post.base import PostBase

__all__ = ['CRTMatchProcessor']


class CRTMatchProcessor(PostBase):
    """Associates TPC particles with CRT hits.
    """

    # Name of the post-processor (as specified in the configuration)
    name = 'crt_match'

    # Alternative allowed names of the post-processor
    aliases = ('run_crt_matching',)

    def __init__(self, crthit_key, obj_type='particle', run_mode='reco'):
        """Initialize the CRT/TPC matching post-processor.
        
        Parameters
        ----------
        crthit_key : str
            Data product key which provides the CRT information
        **kwargs : dict
            Keyword arguments to pass to the CRT-TPC matching algorithm
        """
        # Initialize the parent class
        super().__init__(obj_type, run_mode)

        # Store the relevant attributes
        self.crthit_key = crthit_key

    def process(self, data_dict, result_dict):
        """Find particle/CRT matches for one entry.
        Parameters

        ----------
        data : dict
            Dictionary of data products

        Notes
        -----
        This post-processor also modifies the list of Interactions
        in-place by adding the following attributes:
            particle.is_crthit_matched: bool
                Indicator for whether the given particle has a CRT-TPC match
            particle.crthit_ids: List[int]
                List of IDs for CRT hits that were matched to that particle
        """
        crthits = {}
        assert len(self.crthit_keys) > 0
        for key in self.crthit_keys:
            crthits[key] = data_dict[key]
        
        interactions = result_dict['interactions']
        
        crt_tpc_matches = crt_tpc_manager.get_crt_tpc_matches(int(entry), 
                                                              interactions,
                                                              crthits,
                                                              use_true_tpc_objects=False,
                                                              restrict_interactions=[])

        from matcha.match_candidate import MatchCandidate
        assert all(isinstance(item, MatchCandidate) for item in crt_tpc_matches)

        # crt_tpc_matches is a list of matcha.MatchCandidates. Each MatchCandidate
        # contains a Track and CRTHit instance. The Track class contains the 
        # interaction_id.
        #matched_interaction_ids = [int_id for int_id in crt_tpc_matches.track.interaction_id]
        #matched_interaction_ids = []
        #for match in crt_tpc_matches:
        #    matched_interaction_ids.append(match.track.interaction_id)
        #
        #matched_interactions = [i for i in interactions 
        #                        if i.id in matched_interaction_ids]


        for match in crt_tpc_matches:
            matched_track = match.track
            # To modify the interaction in place, we need to find it in the interactions list
            matched_interaction = None
            for interaction in interactions:
                if matched_track.interaction_id == interaction.id:
                    matched_interaction = interaction
                    break
            matched_crthit = match.crthit
            # Sanity check
            if matched_interaction is None: continue
            matched_interaction.crthit_matched = True
            matched_interaction.crthit_matched_particle_id = matched_track.id
            matched_interaction.crthit_id = matched_crthit.id

            # update_dict['interactions'].append(matched_interaction)
        # update_dict['crt_tpc_matches'].append(crt_tpc_dict)

        return {}, {}
