import numpy as np 

def random_sampling_loc(Batch, Head, Height, Width, count, start_points = [0, 0]):
    def gen_coords():
        x = np.random.randint(0, Height, size=count)
        y = np.random.randint(0, Width, size=count)
        x = x.astype(np.float32) + start_points[0]
        y = y.astype(np.float32) + start_points[1]
        points = np.column_stack((x, y))
    
    def sampling_points(points, radius = 50, num_per_point = 4):
        n = points.shape[0]
        new_points = []
        for (cx, cy) in points:
            for _ in range(num_per_point):
                angle = np.random.uniform(0, 2 * np.pi)
                r = np.random.randint(0, radius + 1)
                dx = int(r * np.cos(angle))
                dy = int(r * np.sin(angle))
                new_points.append([cx + dx, cy + dy])
        return np.array(new_points)

    ret = []
    for _ in range(Batch * Head):
        coords = gen_coords()
        sampling_coords = sampling_points(coords)
        sampling_coords = sampling_coords.reshape(count, 4, 2)
        ret.append(sampling_coords)

    ret = np.asarray(ret).reshape(Batch, Head, count, 1, 4, 2)
    return ret.transpose(0, 2, 1, 3, 4, 5)
