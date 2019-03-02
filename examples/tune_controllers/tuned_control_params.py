controller_params = {
    "DirectTorqueController": (
        # DirectTorqueController,
        {
            "action_scaling": 2.2
        }   
    ),
    # "InverseDynamicsController": (
    #     # InverseDynamicsController,
    #     {
    #         "model_path": "full_kuka_no_collision_no_gravity.xml",
    #         "action_scale": 1.2,
    #         "kp_id": 1.1
    #     }
    # ),
    "RelativeInverseDynamicsController": (
        # RelativeInverseDynamicsController,
        {
            "model_path": "full_kuka_no_collision_no_gravity.xml",
            "action_scale": 2.0,
            "kp_id": 1.1
        }
    ),
    # "PDController": (
    #     # PDController,
    #     {
    #         "kp": 0.27,
    #         "action_scale": 0.4
    #     }
    # ),
    "RelativePDController": (
        # RelativePDController,
        {
            "set_velocity": False,
            "kp": 0.27,
            "action_scale": 0.4
        }
    ),
    # "ImpedanceController": (
    #     # ImpedanceController,
    #     {
    #         "model_path": "full_kuka_no_collision_no_gravity.xml",
    #         "pos_scale": 0.3,
    #         "rot_scale": 0.1,
    #         "stiffness": [0.7, 0.7, 0.7, 2.1, 2.1, 2.1],
    #         "site_name": "ee_site"
    #     }
    # ),
    "ImpedanceControllerV2": (
        # ImpedanceControllerV2,
        {
            "model_path": "full_kuka_no_collision_no_gravity.xml",
            "pos_scale": 0.3,
            "rot_scale": 0.1,
            "stiffness": [2.0, 2.0, 2.0, 6.0, 6.0, 6.0],
            "site_name": "ee_site"
        }
    ),
}