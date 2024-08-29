import pickle
from utils.sim_utils import init_camera_pose
import robosuite.utils.transform_utils as transform_utils 
import numpy as np
from scipy.spatial.transform import Rotation as R

def save_camera_info(env,
                     camera_info_path, 
                     camera_pos, 
                     camera_quat, 
                     scale_factor = 1):
    

    # print("object quat: ", env.obj_quat)
    object_euler = R.from_quat(env.obj_quat).as_euler('xyz', degrees=False)
    # print("object euler: ", object_euler)
    cam_pos_actual = init_camera_pose(env, camera_pos, scale_factor, camera_quat=camera_quat)
    camera_rot_90 = transform_utils.euler2mat(np.array([0, 0, object_euler[0]])) @ transform_utils.quat2mat(camera_quat) 
    camera_quat_rot_90 = transform_utils.mat2quat(camera_rot_90)
    
    camera_euler = R.from_quat(camera_quat).as_euler('xyz', degrees=True)
    camera_euler_for_gamma = np.array([camera_euler[-1], camera_euler[1], -camera_euler[0]]) 
    camera_rot_for_gamma = R.from_euler('xyz', camera_euler_for_gamma, degrees=True).as_matrix()
    camera_rot_for_gamma = transform_utils.euler2mat(np.array([0, 0, object_euler[0]])) @ camera_rot_for_gamma
    camera_quat_for_gamma = R.from_matrix(camera_rot_for_gamma).as_quat()

    camera_config = {'data': {
                        'camera_config': {
                            # 'trans': camera_pos*scale_factor, 
                            'trans': camera_pos, 
                            'quat': camera_quat_rot_90,
                            'trans_absolute': cam_pos_actual,
                            'quat_for_gamma': camera_quat_for_gamma,
                        }
                    }
                } 
    
    with open(camera_info_path, 'wb') as f:
        pickle.dump(camera_config, f) 


def load_wf_grasp_proposals(proposal_path, top_k=10):
    with open(proposal_path, 'rb') as f:
        proposals = np.load(f, allow_pickle=True)
        proposals = proposals['data'].item()
    
    g_pos_wf = proposals['pos']
    g_quat_wf = proposals['quat']
    scores = proposals['scores']

    sorted_grasp_tuples = [(g_pos_wf[i], g_quat_wf[i], scores[i]) for i in range(len(g_pos_wf))]
    sorted_grasp_tuples.sort(key=lambda x: x[2], reverse=True)
    top_k_pos_wf = [g[0] for g in sorted_grasp_tuples][:top_k]
    top_k_quat_wf = [g[1] for g in sorted_grasp_tuples][:top_k]

    return top_k_pos_wf, top_k_quat_wf
    