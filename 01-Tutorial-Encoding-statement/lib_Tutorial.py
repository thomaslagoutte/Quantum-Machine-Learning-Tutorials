import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
####










##### Helper functions for deformation visualization #####

def vector(pt):
    x,y = pt
    return np.array([np.sin(x)*np.cos(y), np.sin(x)*np.sin(y),np.cos(x)])

def angle(A,B,vector=vector):
    return np.arccos(np.dot(vector(A),vector(B)))


def apply_deformations(point, deformation_functions):
    for deformation_function in deformation_functions:
        point = deformation_function(point)
    return point

def local_deformation(pt,deformation_functions,dr=0.01,points=10,vector=vector):
    x,y = pt
    points = 10
    r_deformed = []
    alphas = np.linspace(0, 2*np.pi,points)
    pt_deformed = apply_deformations(pt, deformation_functions)
    for alpha in alphas:
        dx = dr * np.cos(alpha)
        dy = dr * np.sin(alpha)
        B = (x+dx, y+dy)
        B_deformed = apply_deformations(B, deformation_functions)
        angle_AB = angle(pt_deformed,B_deformed,vector=vector)
        r_deformed.append(1-angle_AB /dr)

    return np.var(r_deformed)



def plot_points_colored(points, mapping_functions = None, values=None,
                        cmap='viridis',
                        point_size=40, alpha=0.9,
                        figsize=(6, 6), show_colorbar=True,
                        vmin=None, vmax=None, savepath=None, ax=None , title="plz choose a title it is nicer",
                        just_values=False, predefined_colors=None):
    """
    Plot points and color them according to `func`.
    - Provide either `points` as (N,2) array, or `x` and `y` sequences.
    - `func` can be:
        * vectorized: f(points) -> array-like of length N, or
        * pointwise: f(x_i, y_i) -> scalar (will be called per point).
      If func is None all points are plotted with a single color.
    - If discrete=True the unique values of `func` are mapped to distinct colors.
    Returns the matplotlib Axes.
    """
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be an (N,2) array-like")

    xs, ys = points[:, 0], points[:, 1]

    # Evaluate func
    if values is None and mapping_functions is not None:
        values = [local_deformation(point,mapping_functions) for point in points]

    if just_values:
        return values
    

    ## Plotting ##
    if ax is None:
        fig, (ax, ax3d) = plt.subplots(1,2 , figsize=(figsize[0]*2, figsize[1]), subplot_kw={'projection': None})
        fig.suptitle(title)
    else:
        fig = ax.figure

    if values is None:
        ax.scatter(xs, ys, s=point_size, alpha=alpha, color='C0')
        if show_colorbar:
            pass
    else:
        
        sc = ax.scatter(xs, ys, c=values, cmap=cmap, s=point_size, alpha=alpha,
                        vmin=vmin, vmax=vmax)
        if show_colorbar:
            fig.colorbar(sc, ax=ax)

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # 3D plot of the same points using vector to obtain 3D vectors

    points = [apply_deformations(point, mapping_functions) for point in points]

    vecs = np.array([vector(pt) for pt in points])
    x3, y3, z3 = vecs[:, 0], vecs[:, 1], vecs[:, 2]

    ax3d = fig.add_subplot(1, 2, 2, projection='3d')

    if values is None:
        sc3 = ax3d.scatter(x3, y3, z3, s=point_size, alpha=alpha, color='C0')
    else:
        
        sc3 = ax3d.scatter(x3, y3, z3, c=values, cmap=cmap, s=point_size, alpha=alpha, vmin=vmin, vmax=vmax)
    
    ax3d.set_xlabel('x')
    ax3d.set_ylabel('y')
    ax3d.set_zlabel('z')
    ax3d.set_box_aspect((1, 1, 1))

    ax3d.set_xlim([-1, 1])
    ax3d.set_ylim([-1, 1])
    ax3d.set_zlim([-1, 1])
    plt.tight_layout()
    
    

    if savepath:
        fig.savefig(savepath, bbox_inches='tight')
    return fig , ax, ax3d, values


def Generate_random_points(n_points, range_x=(0, 1), range_y=(0, 1), seed=None):
    if seed is not None:
        np.random.seed(seed)
    x = np.random.uniform(range_x[0], range_x[1], n_points)
    y = np.random.uniform(range_y[0], range_y[1], n_points)
    points = np.column_stack((x, y))
    return points

def Generate_random_clusters_points(n_points, k_clusters, range_x=(0, 1), range_y=(0, 1), seed=None
                                    , init_centers=None):

    if seed is not None:
        np.random.seed(seed)
    points_per_cluster = n_points // k_clusters
    points = []
    k_clusters = len(init_centers) if init_centers is not None else k_clusters
    variance = 0.08
    for i in range(k_clusters):
        if init_centers is not None:
            center_x, center_y = init_centers[i]
        else:
            center_x = np.random.uniform(range_x[0]+0.1, range_x[1]-0.1)
            center_y = np.random.uniform(range_y[0]+0.1, range_y[1]-0.1)
        cluster_points_x = np.random.normal(center_x, variance, points_per_cluster)
        cluster_points_y = np.random.normal(center_y, variance, points_per_cluster)
        cluster_points = np.column_stack((cluster_points_x, cluster_points_y))
        points.append(cluster_points)
    points = np.vstack(points)
    values = [i for i in range(k_clusters) for _ in range(points_per_cluster)]
    # If there are leftover points due to integer division, add random points
    while points.shape[0] < n_points:
        extra_point = np.array([[np.random.uniform(range_x[0], range_x[1]),
                                 np.random.uniform(range_y[0], range_y[1])]])
        points = np.vstack((points, extra_point))
        values.append(k_clusters)  # Assign a new cluster value for extra points
    return points, values

def Generate_circle_points(n_points, center=(0.5, 0.5), radius=0.45, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random angles and radii (sqrt for uniform density)
    theta = np.random.uniform(0, 2*np.pi, n_points)
    r = radius * np.sqrt(np.random.uniform(0, 1, n_points))
    
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    
    return np.column_stack((x, y))

def display_deformation(deformation_functions = [], range_x=(0, 1), range_y=(0, 1),n_points=500, seed=None,
                        point_generation = "random", title=None,
                        **clusters_kwargs):
    
    if point_generation == "random":
        points = Generate_random_points(n_points, range_x, range_y, seed=seed)
        fig, ax, ax3d, values = plot_points_colored(points, mapping_functions=deformation_functions, title=title)
        
    elif point_generation == "clusters":
        points, values = Generate_random_clusters_points(n_points, k_clusters=3, range_x=range_x, range_y=range_y, seed=seed, **clusters_kwargs)
        fig, ax, ax3d, values = plot_points_colored(points, mapping_functions=deformation_functions, values=values, title=title)
        
    elif point_generation == "random_circle":
        # This calls the new function defined in Step 1
        points = Generate_circle_points(n_points, seed=seed)
        fig, ax, ax3d, values = plot_points_colored(points, mapping_functions=deformation_functions, title=title)

    return fig, ax, ax3d, values