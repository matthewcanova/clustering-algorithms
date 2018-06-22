import math
import random
import bisect
import numpy as np
from matplotlib import pyplot as plt


def square_distance(one, two):
    total = 0
    for i in range(len(one)):
        total += math.pow((one[i] - two[i]), 2)
    return math.sqrt(total)


class Cluster:

    def __init__(self, tuple):

        self.points = [tuple]

    def consume_cluster(self, cluster):
        for point in cluster.points:
            self.points.append(point)


class TwoDimCluster:

    def __init__(self, file):

        self.clusters = []

        with open(file) as f:
            for line in f:
                line = line.split()
                line = [float(i) for i in line[1:]]
                tuple(line)
                self.clusters.append(Cluster(line))

    # Returns a list of Clusters which have been hierarchically clustered
    # based on cluster_type until num_clusters Clusters remain
    def cluster(self, num_clusters, cluster_type):

        new_clusters = list(self.clusters)

        while len(new_clusters) > num_clusters:

            if cluster_type == 'Single':
                shortest = self.single(new_clusters[0], new_clusters[1])
            elif cluster_type == 'Complete':
                shortest = self.complete(new_clusters[0], new_clusters[1])
            else:
                shortest = self.mean(new_clusters[0], new_clusters[1])

            point_one = 0
            point_two = 1

            for i in range(len(new_clusters)):
                for j in range(i+1, len(new_clusters)):

                    if cluster_type == 'Single':
                        distance = self.single(new_clusters[i], new_clusters[j])
                    elif cluster_type == 'Complete':
                        distance = self.complete(new_clusters[i], new_clusters[j])
                    else:
                        distance = self.mean(new_clusters[i], new_clusters[j])

                    if distance < shortest:
                        shortest = distance
                        point_one = i
                        point_two = j

            new_clusters[point_one].consume_cluster(new_clusters[point_two])
            del new_clusters[point_two]

        return new_clusters

    def random_center(self, num_centers):
        centers = []

        for i in range(num_centers):
            centers.append(Cluster(random.choice(self.clusters).points[0]))

        return centers

    def lloyds(self, centers):

        clusters = list(self.clusters)
        new_clusters = [Cluster((0, 0)) for i in range(len(centers))]

        while clusters:
            cluster = clusters.pop()
            shortest = self.mean(cluster, centers[0])
            index = 0
            for j in range(1, len(centers)):
                dist = self.mean(cluster, centers[j])
                if dist < shortest:
                    shortest = dist
                    index = j

            new_clusters[index].consume_cluster(cluster)

        for cluster in new_clusters:
            cluster.points.pop()

        return new_clusters

    # Starting from the first Cluster in the list, returns num_centers Clusters
    # that are max distance from all other centers
    def gonzalez(self, num_centers):

        distances = {}
        centers = [self.clusters[0]]

        for i in range(len(self.clusters)):
            distances[self.clusters[i]] = []

        while len(centers) < num_centers:
            for i in range(len(self.clusters)):
                distances[self.clusters[i]].append(square_distance(self.clusters[i].points[0],
                                                                       centers[-1].points[0]))

            biggest_min = 0
            new_center = (0, 0)

            for key, value in distances.items():
                if min(value) > biggest_min:
                    biggest_min = min(value)
                    new_center = key

            centers.append(new_center)

        return centers

    # Starts at a random Cluster and finds num_centers Clusters, which are chosen randomly
    # based on the distance to the nearest center.
    def k_means_plus(self, num_centers):

        first_center = Cluster(random.choice(self.clusters).points[0])
        centers = [first_center]

        while len(centers) < num_centers:

            dists = []

            for i in range(len(self.clusters)):
                shortest = math.pow(square_distance(centers[0].points[0], self.clusters[i].points[0]), 2)
                for j in range(1, len(centers)):
                    dist = math.pow(square_distance(centers[j].points[0], self.clusters[i].points[0]), 2)
                    if dist < shortest:
                        shortest = dist
                dists.append(shortest)

            cum_dists = np.cumsum(dists)

            random_pick = random.random() * cum_dists[len(cum_dists)-1]

            new_center = bisect.bisect_left(cum_dists, random_pick)

            next_center = Cluster(self.clusters[new_center].points[0])

            centers.append(next_center)

        return centers

    def median_cost(self, centers):

        clusters = self.lloyds(centers)

        num_points = 0

        cost = 0

        for i in range(len(clusters)):
            for point in clusters[i].points:
                num_points += 1
                dist = square_distance(point, centers[i].points[0])
                cost += dist

        return cost / num_points

    def mean_cost(self, centers):

        clusters = self.lloyds(centers)

        num_points = 0

        cost = 0

        for i in range(len(clusters)):
            for point in clusters[i].points:
                num_points += 1
                dist = math.pow(square_distance(point, centers[i].points[0]), 2)
                cost += dist

        return math.sqrt(cost / num_points)

    def max_cost(self, centers):

        clusters = self.lloyds(centers)

        max_cost = 0

        for i in range(len(clusters)):
            for point in clusters[i].points:
                dist = square_distance(point, centers[i].points[0])
                if dist > max_cost:
                    max_cost = dist

        return max_cost

    def single(self, cluster_one, cluster_two):
        one = cluster_one.points
        two = cluster_two.points

        shortest = square_distance(one[0], two[0])
        best_one = 0
        best_two = 0

        for i in range(len(one)):
            for j in range(i+1, len(two)):
                dist = square_distance(one[i], two[j])
                if dist < shortest:
                    shortest = dist
                    best_one = i
                    best_two = j

        return square_distance(one[best_one], two[best_two])

    def complete(self, cluster_one, cluster_two):
        one = cluster_one.points
        two = cluster_two.points

        longest = square_distance(one[0], two[0])
        best_one = 0
        best_two = 0

        for i in range(len(one)):
            for j in range(i + 1, len(two)):
                dist = square_distance(one[i], two[j])
                if dist > longest:
                    longest = dist
                    best_one = i
                    best_two = j

        return square_distance(one[best_one], two[best_two])

    def mean(self, cluster_one, cluster_two):
        one_x_total = 0
        one_y_total = 0
        for point in cluster_one.points:
            one_x_total += point[0]
            one_y_total += point[1]
        one_mean = (one_x_total / len(cluster_one.points), one_y_total / len(cluster_one.points))

        two_x_total = 0
        two_y_total = 0
        for point in cluster_two.points:
            two_x_total += point[0]
            two_y_total += point[1]
        two_mean = (two_x_total / len(cluster_two.points), two_y_total / len(cluster_two.points))

        return square_distance(one_mean, two_mean)


# Problem 1A

cluster_one = TwoDimCluster('C1.txt')
results_one_single = cluster_one.cluster(4, 'Single')
results_one_complete = cluster_one.cluster(4, 'Complete')
results_one_mean = cluster_one.cluster(4, 'Mean')

colors = ['r', 'g', 'b', 'k']

color_num = 0
for cluster in results_one_single:
    x = []
    y = []
    for point in cluster.points:
        x.append(point[0])
        y.append(point[1])
    plt.plot(x, y, marker='.', markersize=10, linestyle='None', color=colors[color_num])
    color_num += 1

plt.axis([-10, 9, -5, 5])
plt.title("Hierarchical Clustering of C1.txt using Single-Link Cluster Distance")
plt.xlabel("Dimension 1 (x)")
plt.ylabel("Dimension 2 (y)")

plt.show()

# color_num = 0
# for cluster in results_one_complete:
#     x = []
#     y = []
#     for point in cluster.points:
#         x.append(point[0])
#         y.append(point[1])
#     plt.plot(x, y, marker='.', markersize=10, linestyle='None', color=colors[color_num])
#     color_num += 1
#
# plt.axis([-10, 9, -5, 5])
# plt.title("Hierarchical Clustering of C1.txt using Complete-Link Cluster Distance")
# plt.xlabel("Dimension 1 (x)")
# plt.ylabel("Dimension 2 (y)")
#
# plt.show()
#
#
# color_num = 0
# for cluster in results_one_mean:
#     x = []
#     y = []
#     for point in cluster.points:
#         x.append(point[0])
#         y.append(point[1])
#     plt.plot(x, y, marker='.', markersize=10, linestyle='None', color=colors[color_num])
#     color_num += 1
#
# plt.axis([-10, 9, -5, 5])
# plt.title("Hierarchical Clustering of C1.txt using Mean-Link Cluster Distance")
# plt.xlabel("Dimension 1 (x)")
# plt.ylabel("Dimension 2 (y)")
#
# plt.show()


# Problem 2A Gonzalez

# cluster_two = TwoDimCluster('C2.txt')
# cluster_two_gonz_centers = cluster_two.gonzalez(3)
# cluster_two_gonz_clustered = cluster_two.lloyds(cluster_two_gonz_centers)
#
# center_tuples = []
# for cluster in cluster_two_gonz_centers:
#     center_tuples.append(cluster.points[0])
#
# colors = ['k', 'g', 'b', 'r']
#
# color_num = 0
#
# for cluster in cluster_two_gonz_clustered:
#     x = []
#     y = []
#     for point in cluster.points:
#             x.append(point[0])
#             y.append(point[1])
#     plt.plot(x, y, markersize=9, marker='.', linestyle='None', color=colors[color_num])
#     color_num += 1
#
# for point in center_tuples:
#     plt.plot(point[0], point[1], markersize=9, marker='o', markeredgecolor='r', markeredgewidth=2, markerfacecolor='None')
#
# print('Max Cost For Gonzalez is: ' + str(cluster_two.max_cost(cluster_two_gonz_centers)))
# print('Mean Cost For Gonzalez is: ' + str(cluster_two.mean_cost(cluster_two_gonz_centers)))
#
# plt.title("Clustering of C2.txt using the Gonzalez algorithm to find k-means")
# plt.xlabel("Dimension 1 (x)")
# plt.ylabel("Dimension 2 (y)")
#
# plt.show()

# Problem 2A k-means++

# cluster_two = TwoDimCluster('C2.txt')
# means = []
# avg_max_cost = 0
# avg_mean_cost = 0
#
#
# for i in range(100):
#     cluster_two_plus_centers = cluster_two.k_means_plus(3)
#     mean_cost = cluster_two.mean_cost(cluster_two_plus_centers)
#     avg_max_cost += cluster_two.max_cost(cluster_two_plus_centers)
#     avg_mean_cost += mean_cost
#     means.append(mean_cost)
#
# means.sort()
#
# # x axis from lowest sample value to highest
# x_axis = means
#
# # y axis values
# y_axis = []
#
# # calculate y axis values
# for i in x_axis:
#     count = 0
#     for j in means:
#         if j <= i:
#             count += 1
#     y_axis.append(count / len(means))
#
# plt.plot(x_axis, y_axis)
#
# print('Average Max Cost For k-Means++ is: ' + str(avg_max_cost/100))
# print('Average Mean Cost For k-Means++ is: ' + str(avg_mean_cost/100))
#
# plt.title("CDF of the Mean Cost for the k-Means++ algorithm Clusters")
# plt.xlabel("Mean Cost (x)")
# plt.ylabel("P(Mean Cost <= x) (y)")
#
# plt.show()

# Problem 2B

#cluster_two = TwoDimCluster('C2.txt')

# Naive

# centers = []
#
# for i in range(0, 3):
#     centers.append(cluster_two.clusters[i])
#
# center_tuples = []
# for cluster in centers:
#     center_tuples.append(cluster.points[0])
#
# clusters = cluster_two.lloyds(centers)
#
# colors = ['k', 'g', 'b', 'r']
#
# color_num = 0
#
# for cluster in clusters:
#     x = []
#     y = []
#     for point in cluster.points:
#             x.append(point[0])
#             y.append(point[1])
#     plt.plot(x, y, markersize=9, marker='.', linestyle='None', color=colors[color_num])
#     color_num += 1
#
# for point in center_tuples:
#     plt.plot(point[0], point[1], markersize=9, marker='o', markeredgecolor='r', markeredgewidth=2, markerfacecolor='None')
#
# print('3-Means Cost For Naive Centers is: ' + str(cluster_two.mean_cost(centers)))
#
# plt.title("Clustering with Lloyd's Algorithm using a naive choice of centers")
# plt.xlabel("Dimension 1 (x)")
# plt.ylabel("Dimension 2 (y)")
#
# plt.show()


# Gonzalez

# centers = cluster_two.gonzalez(3)
#
# center_tuples = []
# for cluster in centers:
#     center_tuples.append(cluster.points[0])
#
# clusters = cluster_two.lloyds(centers)
#
# colors = ['k', 'g', 'b', 'r']
#
# color_num = 0
#
# for cluster in clusters:
#     x = []
#     y = []
#     for point in cluster.points:
#             x.append(point[0])
#             y.append(point[1])
#     plt.plot(x, y, markersize=9, marker='.', linestyle='None', color=colors[color_num])
#     color_num += 1
#
# for point in center_tuples:
#     plt.plot(point[0], point[1], markersize=9, marker='o', markeredgecolor='r', markeredgewidth=2, markerfacecolor='None')
#
# print('3-Means Cost For Gonzalez Centers is: ' + str(cluster_two.mean_cost(centers)))
#
# plt.title("Clustering with Lloyd's Algorithm using Gonzalez for choice of centers")
# plt.xlabel("Dimension 1 (x)")
# plt.ylabel("Dimension 2 (y)")
#
# plt.show()


# k-means++

# means = []
# avg_mean_cost = 0
#
#
# for i in range(100):
#     cluster_two_plus_centers = cluster_two.k_means_plus(3)
#     mean_cost = cluster_two.mean_cost(cluster_two_plus_centers)
#     avg_mean_cost += mean_cost
#     means.append(mean_cost)
#
# means.sort()
#
# # x axis from lowest sample value to highest
# x_axis = means
#
# # y axis values
# y_axis = []
#
# # calculate y axis values
# for i in x_axis:
#     count = 0
#     for j in means:
#         if j <= i:
#             count += 1
#     y_axis.append(count / len(means))
#
# plt.plot(x_axis, y_axis)
#
# print('Average Mean Cost For k-Means++ is: ' + str(avg_mean_cost/100))
#
# plt.title("CDF of the Mean Cost for the k-Means++ with Lloyd's")
# plt.xlabel("Mean Cost (x)")
# plt.ylabel("P(Mean Cost <= x) (y)")
#
# plt.show()




# Problem 3

# def ball_volume(dim, radius):
#     num = math.pow(math.pi, dim/2)
#     if dim == 3:
#         den = 1.33
#     else:
#         den = math.factorial(dim/2)
#     return (num/den) * (math.pow(radius, dim))
#
#
# def box_volume(dim, radius):
#     return math.pow(2*radius, dim)
#
# # 3-1, d = 2
#
# dim = 2
# ball_radius = 1
# box = box_volume(dim, 1)
# ball = ball_volume(dim, ball_radius)
#
# while box > ball:
#     ball_radius += 0.001
#     ball = ball_volume(dim, ball_radius)
#
# print(ball_radius)
#
# # 3-2, d = 3
#
# dim = 3
# ball_radius = 1
# box = box_volume(dim, 1)
# ball = ball_volume(dim, ball_radius)
#
# while box > ball:
#     ball_radius += 0.001
#     ball = ball_volume(dim, ball_radius)
#
# print(ball_radius)
#
# # 3-3, d = 4
#
# dim = 4
# ball_radius = 1
# box = box_volume(dim, 1)
# ball = ball_volume(dim, ball_radius)
#
# while box > ball:
#     ball_radius += 0.001
#     ball = ball_volume(dim, ball_radius)
#
# print(ball_radius)
#
#
# 3-4
#
# # box / ball
# #
# # box = 2r^d
# #
# # ball = pi^(d/2) / ((d/2)!) * r^d
# #
# # r^d cancels out, we are left with 2 / (pi^(d/2) / ((d/2)!))
#
# 3-5
#
# dim = 2
#
# x = []
# y = []
#
# while dim < 21:
#     ball_radius = 1
#     box = box_volume(dim, 1)
#     ball = ball_volume(dim, ball_radius)
#
#     while box > ball:
#         ball_radius += 0.001
#         ball = ball_volume(dim, ball_radius)
#
#     x.append(dim)
#     y.append(ball_radius)
#     dim += 2
#
# plt.plot(x, y, marker='.', linestyle='-')
# plt.title("Expansion Factor of c as d increases")
# plt.xlabel("Dimensions")
# plt.ylabel("Expansion Factor (c)")
#
# plt.show()


#4

cluster_two = TwoDimCluster('C3.txt')


# http://stackoverflow.com/questions/6283080/random-unit-vector-in-multi-dimensional-space
def make_rand_vector(dims):
    vec = [random.gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return [x/mag for x in vec]

latest_result = 1

while True:

    centers_set = []
    fitnesses = []

    for k in range(30):
        centers = cluster_two.random_center(4)
        centers_set.append(centers)
        fitnesses.append(1 / cluster_two.median_cost(centers))

    best_fitness = max(fitnesses)

    best_fitness_record = 0

    progress_made = True

    while progress_made:

        cum_fitnesses = np.cumsum(fitnesses)

        best_fitness_record += 1

        if best_fitness_record > 3000:
            if (max(fitnesses) - best_fitness) < 0.001:
                progress_made = False
            else:
                best_fitness_record = 0
                best_fitness = max(fitnesses)

        random_pick = random.random() * cum_fitnesses[len(cum_fitnesses) - 1]
        agent_one_index = bisect.bisect_left(cum_fitnesses, random_pick)

        random_pick = random.random() * cum_fitnesses[len(cum_fitnesses) - 1]
        agent_two_index = bisect.bisect_left(cum_fitnesses, random_pick)

        while centers_set[agent_two_index] == centers_set[agent_one_index]:
            random_pick = random.random() * cum_fitnesses[len(cum_fitnesses) - 1]
            agent_two_index = bisect.bisect_left(cum_fitnesses, random_pick)

        child_centers = []

        # for each center in the 4-center set
        for i in range(len(centers_set[agent_one_index])):
            # Find the closest center in the other set
            shortest = square_distance(centers_set[agent_one_index][i].points[0], centers_set[agent_two_index][0].points[0])
            mate = 0
            for j in range(1, len(centers_set[agent_two_index])):
                dist = square_distance(centers_set[agent_one_index][i].points[0], centers_set[agent_two_index][j].points[0])
                if dist < shortest:
                    shortest = dist
                    mate = j

            # mate the two centers: agent_one[i] and agent_two[mate]
            one = centers_set[agent_one_index][i].points[0]
            two = centers_set[agent_two_index][mate].points[0]

            child = []

            for k in range(len(one)):
                child.append(random.uniform(one[k], two[k]))

            rand_vect = make_rand_vector(5)
            rand_vect = [i * (shortest * random.random() * 1.618) for i in rand_vect]
            child = [child[i] + rand_vect[i] for i in range(len(child))]
            child_centers.append(Cluster(tuple(child)))

        child_fitness = 1 / cluster_two.median_cost(child_centers)

        lowest_index = fitnesses.index(min(fitnesses))

        if child_fitness > fitnesses[lowest_index]:
            centers_set[lowest_index] = child_centers
            fitnesses[lowest_index] = child_fitness

    loc = fitnesses.index(max(fitnesses))
    winner = centers_set[loc]
    latest_result = cluster_two.median_cost(winner)
    for center in winner:
        print('Center: ' + str(center.points[0]))
    print('Center set had a final cost of: ' + str(latest_result) + '\n')

