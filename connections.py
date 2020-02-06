import random
import math



def calc_distance(pa, pb):
    distance = math.sqrt( (pa[0] - pb[0])**2.0 + (pa[1] - pb[1])**2.0 )
    return distance

def is_allowed(pair, index_a_max, index_b_max):
    allowed = False
    if pair[0] >= 0 and pair[0] < index_a_max:
        if pair[1] >= 0 and pair[1] < index_b_max:
            allowed = True
    return allowed

def get_neighbor_cell_indices(index_a, index_b, search_distance, index_a_max, index_b_max):
    if search_distance == 0:
        return [ (index_a, index_b) ]
    else:
        neighbor_cell_indices = []

        for k in range(-search_distance,search_distance+1):
            neighbor_cell_indices.append( (index_a+k, index_b+search_distance) )
            neighbor_cell_indices.append( (index_a+k, index_b-search_distance) )

        for k in range(-search_distance+1,search_distance):
            neighbor_cell_indices.append( (index_a+search_distance, index_b+k) )
            neighbor_cell_indices.append( (index_a-search_distance, index_b+k) )

        neighbor_cell_indices_filtered = []
        for pair in neighbor_cell_indices:
            if is_allowed(pair, index_a_max, index_b_max):
                neighbor_cell_indices_filtered.append(pair)

        return neighbor_cell_indices_filtered


def get_grid_with_points(nx, ny, points, xmax, ymax):
    cell_width_x = xmax / float(nx)
    cell_width_y = ymax / float(ny)
    grid = [ [ [] for k in range(ny) ] for i in range(nx) ]
    for p in points:
        i = int(p[0] / cell_width_x)
        k = int(p[1] / cell_width_y)
        grid[i][k].append(p)
    initial_i, initial_k = i, k
    return grid, initial_i, initial_k

def get_neighboring_points(points, max_distance, xmax, ymax):
    """
    Sort the points such that each point
    is followed by its closest neighbor.
    This is needed to know which points should be connected with
    line segments.
    Only if there is no neighbor within the circle with the
    radius max_distance, any other point may follow.
    """
    nx = int(xmax / max_distance) + 1
    ny = int(ymax / max_distance) + 1
    grid, initial_i, initial_k = get_grid_with_points(nx, ny, points, xmax, ymax)

    current_i = initial_i
    current_k = initial_k
    current_point_index = 0
    current_point = grid[current_i][current_k][current_point_index]
    del grid[current_i][current_k][current_point_index]
    connected = [current_point]

    for i in range(len(points)-1):
        found = False
        current_best_cell_index_i = None
        current_best_cell_index_k = None
        current_best_point_index = None
        current_best_distance = None

        cell_distance = 1
        neighbor_cell_indices = [(current_i, current_k)] + get_neighbor_cell_indices(current_i, current_k, cell_distance, nx, ny)
        for cell_i, cell_k in neighbor_cell_indices:
            for point_index, point in enumerate(grid[cell_i][cell_k]):
                distance = calc_distance(current_point, point)
                if current_best_distance == None or distance < current_best_distance:
                    current_best_distance = distance
                    current_best_cell_index_i = cell_i
                    current_best_cell_index_k = cell_k
                    current_best_point_index = point_index
        if current_best_distance != None:
            found = True

        if found:
            current_i = current_best_cell_index_i
            current_k = current_best_cell_index_k
            current_point_index = current_best_point_index
            current_point = grid[current_i][current_k][current_point_index]
            connected.append(current_point)
            # delete the found point
            del grid[current_i][current_k][current_point_index]
        else:
            # get any other point
            found_new = False
            cell_distance = 2
            while not found_new:
                neighbor_cell_indices = get_neighbor_cell_indices(current_i, current_k, cell_distance, nx, ny)
                random.shuffle(neighbor_cell_indices)
                for cell_i, cell_k in neighbor_cell_indices:
                    if grid[cell_i][cell_k] != []:
                        found_new = True
                        current_i = cell_i
                        current_k = cell_k
                        current_point_index = 0
                        current_point = grid[current_i][current_k][current_point_index]
                        connected.append(current_point)
                        # delete the found point
                        del grid[current_i][current_k][current_point_index]
                        break

                cell_distance += 1

    return connected
