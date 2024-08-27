import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 
from sklearn.decomposition import PCA 
from scipy.optimize import minimize


def project_points_onto_plane(points, normal_vector, origin):
    projected_points = [] 
    for p in points:
        v = p - origin 
        distance = np.dot(v, normal_vector)
        projected_point = p - distance * normal_vector
        projected_points.append(projected_point)
    return np.array(projected_points)


class InteractivePerception:
    '''
    Estimates the axis and type of the joint given a trajectory of the end effector.
    '''
    def __init__(self, trajectory, logvar=0.1):
        self.trajectory = trajectory 
        self.prismatic_model = LinearRegression()

    def prismatic_error(self):
        X = self.trajectory[:, 1:] 
        y = self.trajectory[:, 0]   
        self.prismatic_model.fit(X, y)

        # Predict and calculate error
        y_pred = self.prismatic_model.predict(X)
        mse = mean_squared_error(y, y_pred)

        return mse

    # def revolute_error(self):
    #     mean = np.mean(self.trajectory, axis=0) # mean is assumed to be on the plane
    #     centered_points = self.trajectory - mean

    #     pca = PCA(n_components=3) 
    #     pca.fit(centered_points)

    #     normal_vector = pca.components_[2] 
    #     plane_vectors = pca.components_[:2]
    #     # plane_vectors += mean 

    #     print('estimated axis direction: ', normal_vector) 
    #     print('estimated plane: ', plane_vectors) 

    #     d_squared = np.sum(np.dot((self.trajectory-mean), normal_vector)**2)

    #     # projected_points = project_points_onto_plane(self.trajectory, normal_vector, mean_projection) 

    def project_points_onto_plane(self, points, normal, point_on_plane):
        projected_points = []
        for p in points:
            vector = p - point_on_plane
            distance = np.dot(vector, normal)
            projection = p - distance * normal
            projected_points.append(projection)
        return np.array(projected_points)

    # Step 2: Define the circle fitting function in 3D
    def fit_circle_to_arc(self, points_3d):
        # Function to fit a circle to 3D points projected onto a 2D plane
        pca = PCA(n_components=2)
        points_2d = pca.fit_transform(points_3d)
        def calc_R(c):
            """ Calculate the distance of each 2D point from the center c=(xc, yc) """
            return np.sqrt((points_2d[:, 0] - c[0]) ** 2 + (points_2d[:, 1] - c[1]) ** 2)

        def loss_function(c):
            """ Calculate the loss as the sum of squared differences from the mean radius """
            Ri = calc_R(c)
            return np.sum((Ri - Ri.mean()) ** 2)

        # Perform PCA to project the 3D points onto a 2D plane

        # A rough initial estimate for the circle center could be the centroid of the arc points
        center_estimate = np.mean(points_2d, axis=0)
        result = minimize(loss_function, center_estimate)
        circle_center_2d = result.x
        radius = calc_R(circle_center_2d).mean()

        # Map the 2D center back to 3D using the PCA components
        circle_center_3d = pca.inverse_transform(circle_center_2d)

        return circle_center_3d, radius


    def revolute_error(self):
        mean = np.mean(self.trajectory, axis=0) 
        X = self.trajectory - mean
        pca = PCA(n_components=3)
        pca.fit(X)

        # Normal vector and point on the plane
        normal_vector = pca.components_[-1]
        point_on_plane = np.mean(X, axis=0)

        # Project points onto the plane
        projected_points = project_points_onto_plane(X, normal_vector, point_on_plane)

        # Step 4: Fit the circle to the projected points
        circle_center_3d, radius = self.fit_circle_to_arc(projected_points)

        print("Estimated Circle Center in 3D:", circle_center_3d)
        print("Estimated Radius:", radius) 
        # error = np.sum((self.trajectory - circle_center_3d)**2) 
        e2 = 0
        for i in range(len(self.trajectory)):
            e2 += np.log(np.sqrt(np.abs((np.sum((self.trajectory[i] - circle_center_3d)**2) - radius**2))))
            # e2 += np.log(np.sqrt(np.sum((self.trajectory[i] - circle_center_3d)**2) - radius**2))
        e2 /= len(self.trajectory)
        return e2


if __name__ == '__main__': 
    angles = np.linspace(0, 2*np.pi, 1000) 
    radius = 10
    trajectory = np.zeros((1000, 3)) 
    trajectory = np.random.randn(1000, 3)
    trajectory[:, 0] += np.cos(angles) * radius 
    trajectory[:, 1] += np.sin(angles) * radius
    ip = InteractivePerception(trajectory)
    error = ip.revolute_error() 
    print(trajectory)
    print('revolute error: ', error) 
    pris_error = ip.prismatic_error() 
    print('prismatic error: ', pris_error)