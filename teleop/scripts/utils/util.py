import numpy as np
import tf.transformations as tft

# Define the two transformation matrices
# T_psm1_to_cart = np.array([[ 0.7983,  0.5931,  0.105 , -0.8011],
#                             [ 0.5626, -0.7965,  0.2218,  0.775 ],
#                             [ 0.2152, -0.118 , -0.9694,  0.9663],
#                             [ 0.    ,  0.    ,  0.    ,  1.    ]])

T_psm1_to_cart = np.array([[ 4.8074e-01,  8.3056e-01,  2.8117e-01, -1.4000e-01],
                            [ 8.7687e-01, -4.5535e-01, -1.5414e-01,  1.2382e+00],
                            [ 4.0908e-06,  3.2065e-01, -9.4720e-01,  1.0802e+00],
                            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])



# T_world_to_tool = np.array([[ 4.8074e-01,  8.5006e-01,  2.1516e-01, -1.5669e-01],
#                             [ 8.7687e-01, -4.6604e-01, -1.1795e-01,  1.3432e+00],
#                             [ 1.2441e-05,  2.4537e-01, -9.6943e-01,  9.5586e-01],
#                             [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])

T_world_to_tool = np.array([[ 4.8074e-01,  8.3056e-01,  2.8117e-01, -1.4000e-01],
                            [ 8.7687e-01, -4.5535e-01, -1.5414e-01,  1.2382e+00],
                            [ 4.0908e-06,  3.2065e-01, -9.4720e-01,  1.0802e+00],
                            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])




# Multiply the matrices
result_matrix = T_psm1_to_cart@T_world_to_tool

# Extract translation
translation = result_matrix[0:3, 3]

# Extract rotation matrix
rotation_matrix = result_matrix[0:3, 0:3]

# Convert rotation matrix to roll, pitch, yaw
qx,qy,qz,qw = tft.quaternion_from_matrix(result_matrix)

# Print the result
print("Translation: x = {:.4f}, y = {:.4f}, z = {:.4f}".format(translation[0], translation[1], translation[2]))
print("Rotation: qx = {:.4f}, qy = {:.4f}, qz = {:.4f}, qw = {:.4f}".format(qx,qy,qz,qw))