import io
from PIL import Image

import numpy as np
import transformations as tfs

import matplotlib.pyplot as plt
import seaborn as sns

def get_best_yaw(C):
    '''
    maximize trace(Rz(theta) * C)
    '''
    assert C.shape == (3, 3)

    A = C[0, 1] - C[1, 0]
    B = C[0, 0] + C[1, 1]
    theta = np.pi / 2 - np.arctan2(B, A)

    return theta

def rot_z(theta):
    R = tfs.rotation_matrix(theta, [0, 0, 1])
    R = R[0:3, 0:3]

    return R

def align_umeyama(model, data, known_scale=False, yaw_only=False):
    """Implementation of the paper: S. Umeyama, Least-Squares Estimation
    of Transformation Parameters Between Two Point Patterns,
    IEEE Trans. Pattern Anal. Mach. Intell., vol. 13, no. 4, 1991.

    model = s * R * data + t

    Input:
    model -- first trajectory (nx3), numpy array type
    data -- second trajectory (nx3), numpy array type

    Output:
    s -- scale factor (scalar)
    R -- rotation matrix (3x3)
    t -- translation vector (3x1)
    t_error -- translational error per point (1xn)

    """

    # substract mean
    mu_M = model.mean(0)
    mu_D = data.mean(0)
    model_zerocentered = model - mu_M
    data_zerocentered = data - mu_D
    n = np.shape(model)[0]

    # correlation
    C = 1.0/n*np.dot(model_zerocentered.transpose(), data_zerocentered)
    sigma2 = 1.0/n*np.multiply(data_zerocentered, data_zerocentered).sum()
    U_svd, D_svd, V_svd = np.linalg.linalg.svd(C)
    D_svd = np.diag(D_svd)
    V_svd = np.transpose(V_svd)

    S = np.eye(3)
    if(np.linalg.det(U_svd)*np.linalg.det(V_svd) < 0):
        S[2, 2] = -1

    if yaw_only:
        rot_C = np.dot(data_zerocentered.transpose(), model_zerocentered)
        theta = get_best_yaw(rot_C)
        R = rot_z(theta)
    else:
        R = np.dot(U_svd, np.dot(S, np.transpose(V_svd)))

    if known_scale:
        s = 1
    else:
        s = 1.0/sigma2*np.trace(np.dot(D_svd, S))

    t = mu_M-s*np.dot(R, mu_D)

    return s, R, t

def compute_absolute_error_translation(p_gt, p_es_aligned):

    e_trans_vec = (p_gt - p_es_aligned)
    
    # Compute Euclidean distances
    errors = np.linalg.norm(e_trans_vec, axis=1)
    
    # Compute RMSE
    ate = np.sqrt(np.mean(errors ** 2))

    return ate

def plot_trajectories3D(traj1, traj2, filename='trajectories.png'):
    """
    Plot two 3D trajectories with line segments between corresponding poses
    and save the plot as an image.
    
    traj1: First trajectory (N, 3)
    traj2: Second trajectory (N, 3)
    filename: The name of the file where the plot will be saved
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the first trajectory
    ax.plot(traj1[:, 0], traj1[:, 1], traj1[:, 2], label='GT', color='red')
    
    # Plot the second trajectory
    ax.plot(traj2[:, 0], traj2[:, 1], traj2[:, 2], label='Estimate', color='blue')
    
    # Draw line segments between corresponding poses
    for i in range(traj1.shape[0]):
        ax.plot([traj1[i, 0], traj2[i, 0]],
                [traj1[i, 1], traj2[i, 1]],
                [traj1[i, 2], traj2[i, 2]], color='gray', linestyle='dotted')

    # Add labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    # Save the plot as an image file
    # plt.savefig(filename)
    # plt.show ()

    return fig

def plot_trajectories2D_legacy(traj1, traj2, filename='trajectories.png'):
    """
    Plot two 2D subplots for x-y and x-z views of the trajectories with line segments between corresponding poses
    and save the plot as an image.
    
    traj1: First trajectory (N, 3)
    traj2: Second trajectory (N, 3)
    filename: The name of the file where the plot will be saved
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Plot the first trajectory (x-y view)
    axes[0].plot(traj1[:, 0], traj1[:, 1], label='GT', color='red')
    axes[0].plot(traj2[:, 0], traj2[:, 1], label='Estimate', color='blue')

    # Draw line segments between corresponding poses (x-y view)
    for i in range(traj1.shape[0]):
        axes[0].plot([traj1[i, 0], traj2[i, 0]],
                     [traj1[i, 1], traj2[i, 1]], color='gray', linestyle='dotted')

    # Add labels and legend for x-y view
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].legend()
    axes[0].set_title('X-Y View')
    axes[0].grid(True)

    # Plot the second trajectory (x-z view)
    axes[1].plot(traj1[:, 0], traj1[:, 2], label='GT', color='red')
    axes[1].plot(traj2[:, 0], traj2[:, 2], label='Estimate', color='blue')

    # Draw line segments between corresponding poses (x-z view)
    for i in range(traj1.shape[0]):
        axes[1].plot([traj1[i, 0], traj2[i, 0]],
                     [traj1[i, 2], traj2[i, 2]], color='gray', linestyle='dotted')

    # Add labels and legend for x-z view
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')
    axes[1].legend()
    axes[1].set_title('X-Z View')
    axes[1].grid(True)

    # Save the plot as an image file
    # plt.savefig(filename)
    # plt.show()

    return fig

def plot_trajectories2D(traj1, traj2, filename='trajectories.png'):
    """
    Plot two 2D subplots for x-y and x-z views of the trajectories with line segments between corresponding poses,
    and save the plot as an image. The estimated trajectory is color-coded based on ATE.
    
    traj1: First trajectory (N, 3)
    traj2: Second trajectory (N, 3)
    filename: The name of the file where the plot will be saved
    """
    sns.set_theme()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Calculate ATE for each pose
    ATE = np.linalg.norm(traj1 - traj2, axis=1)
    
    # Normalize ATE for colormap
    norm = plt.Normalize(ATE.min(), ATE.max())
    cmap = plt.get_cmap('rainbow')
    
    # Plot the first trajectory (x-y view)
    axes[0].plot(traj1[:, 0], traj1[:, 1], label='GT', color='black', linestyle='--')
    
    # Plot the second trajectory (x-y view) with color coding
    for i in range(traj1.shape[0] - 1):
        axes[0].plot(traj2[i:i+2, 0], traj2[i:i+2, 1], color=cmap(norm(ATE[i])))
    
    # Draw line segments between corresponding poses (x-y view)
    for i in range(traj1.shape[0]):
        axes[0].plot([traj1[i, 0], traj2[i, 0]],
                     [traj1[i, 1], traj2[i, 1]], color='gray', linestyle='dotted')

    # Add labels, legend, and grid for x-y view
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].legend()
    axes[0].set_title('X-Y View')
    axes[0].grid(True)

    # Plot the second trajectory (x-z view)
    axes[1].plot(traj1[:, 0], traj1[:, 2], label='GT', color='black', linestyle='--')
    
    # Plot the second trajectory (x-z view) with color coding
    for i in range(traj1.shape[0] - 1):
        axes[1].plot(traj2[i:i+2, 0], traj2[i:i+2, 2], color=cmap(norm(ATE[i])))
    
    # Draw line segments between corresponding poses (x-z view)
    for i in range(traj1.shape[0]):
        axes[1].plot([traj1[i, 0], traj2[i, 0]],
                     [traj1[i, 2], traj2[i, 2]], color='gray', linestyle='dotted')

    # Add labels, legend, and grid for x-z view
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')
    axes[1].legend()
    axes[1].set_title('X-Z View')
    axes[1].grid(True)

    # Add a colorbar to indicate ATE values
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.025, pad=0.04, label='ATE')

    # Set background color
    # fig.patch.set_facecolor('lightgray')
    # for ax in axes:
    #     ax.set_facecolor('lightgray')
        
    # Save the plot as an image file
    # plt.savefig(filename)
    # plt.show()

    return fig

def fig_to_array(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img)
    buf.close()
    return img_array

if __name__ == "__main__":

    def generate_random_similarity_transformation():
        """
        Generate a random similarity transformation consisting of scaling factor s, 
        rotation matrix R, and translation vector t.
        
        Returns:
            s: Scaling factor
            R: Rotation matrix (3, 3)
            t: Translation vector (3,)
        """
        # Random scaling factor
        s = np.random.uniform(0.5, 2.0)

        # Random rotation matrix using axis-angle representation
        theta = np.random.uniform(0, 2 * np.pi)
        axis = np.random.normal(size=3)
        axis = axis / np.linalg.norm(axis)
        R = np.eye(3)
        R += np.sin(theta) * np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R += (1 - np.cos(theta)) * (np.outer(axis, axis) - np.eye(3))

        # Random translation vector
        t = np.random.uniform(-1, 1, size=3)

        return s, R, t

    def generate_trajectories(N, noise_level=0.1, scale_factor=1):
        """
        Generate two sample trajectories stored in N x 3 numpy arrays.
        
        N: Number of points in the trajectory
        noise_level: Standard deviation of the Gaussian noise to add to the second trajectory
        
        Returns:
            traj1: First trajectory (N, 3)
            traj2: Second trajectory (N, 3) with added noise
        """
        # Generate a basic trajectory (e.g., a straight line with some variations)
        t = np.linspace(0, 1, N)
        traj1 = np.stack([t, np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)], axis=1)
        
        traj2 = traj1

        # Generate the second trajectory by adding some Gaussian noise
        # traj2 = traj2 + np.random.normal(scale=noise_level, size=traj1.shape)

        # scale the original trajectory
        # traj2 = traj2 * scale_factor

        # apply random similarity transformation upon the original trajectory
        s, R, t = generate_random_similarity_transformation()
        traj2 = s * (traj2 @ R.T) + t
        
        return traj1, traj2
    
    # generate two trajectories
    N = 10
    traj1, traj2 = generate_trajectories(N, noise_level=0.1, scale_factor=2)
    # plot them before alignment
    plot_trajectories3D(traj1, traj2, filename="trajs_before_align.png")
    
    # align w/ SIM3
    s, R, t = align_umeyama(traj1, traj2)

    # apply transformation on the data (traj2)
    traj2_aligned = (s * (R @ traj2.T)).T + t
    
    # plot them post alignment
    plot_trajectories3D(traj1, traj2_aligned, filename="trajs_after_align.png")

    e_trans, _ = compute_absolute_error_translation(traj2_aligned, traj1)
    print(f"ATE for translation vector is {e_trans}")
