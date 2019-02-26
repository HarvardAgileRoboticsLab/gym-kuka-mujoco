class ManualImpedancePegInsertionPolicy:
    def predict(self, obs, *args, **kwargs):
        dframe = obs[14:20].copy()
        # import pdb; pdb.set_trace()
        action = dframe
        action[3:6] = action[3:6]
        return action, None
