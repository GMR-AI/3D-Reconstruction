import numpy as np

def toUnhomogenize(pts):
    return pts[:, :-1] / pts[:, -1:]

def triangulate(pts1, pts2, projectionMatrix1, projectionMatrix2):
    N = min(len(pts1), len(pts2))
    X = np.zeros((N, 4))

    for i in range(N):
        A = np.vstack((
            pts1[i, 1] * projectionMatrix1[2, :] - projectionMatrix1[1, :],
            projectionMatrix1[0, :] - pts1[i, 0] * projectionMatrix1[2, :],
            pts2[i, 1] * projectionMatrix2[2, :] - projectionMatrix2[1, :],
            projectionMatrix2[0, :] - pts2[i, 0] * projectionMatrix2[2, :]
        ))

        _, _, VT = np.linalg.svd(A)
        X[i, :] = VT[-1, :]

    X = X.T
    X /= X[-1, :]

    pts1_reprojected = projectionMatrix1 @ X
    pts2_reprojected = projectionMatrix2 @ X

    pts1_reprojected /= pts1_reprojected[-1, :]
    pts2_reprojected /= pts2_reprojected[-1, :]

    pts1_homo = np.vstack((pts1.T, np.ones((1, len(pts1)))))
    pts2_homo = np.vstack((pts2.T, np.ones((1, len(pts2)))))

    err1 = np.sum((pts1_reprojected - pts1_homo)**2)
    err2 = np.sum((pts2_reprojected - pts2_homo)**2)

    err = err1 + err2
    X = toUnhomogenize(X.T)
    return X, err

