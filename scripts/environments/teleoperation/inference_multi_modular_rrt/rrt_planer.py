import numpy as np
import random

OBJECT_RADIUS = 0.05
TARGET_X_MIN = 0.78
TARGET_X_MAX = 0.80
TARGET_Y_MIN = -0.1
TARGET_Y_MAX = 0.1

class Node:
    def __init__(self, position):
        """
        Node for RRT tree
        Args:
            position: [x, y] coordinates in 2D space
        """
        self.position = np.array(position)  # 2D position
        self.parent = None
        self.cost = 0.0

class RRT:
    def __init__(self, q_start, q_goal, obstacles, robot_radius=0.04, 
                 q_limits=None, max_iter=10000, step_size=0.5):
        """
        RRT Initialization for 2D navigation
        
        Parameters:
        - q_start: Starting position [x, y]
        - q_goal: Goal position [x, y]
        - obstacles: List of obstacle tuples (x, y, radius)
        - robot_radius: Robot collision radius
        - q_limits: [(min_x, max_x), (min_y, max_y)] workspace bounds
        - max_iter: Maximum number of iterations
        - step_size: Maximum step size to expand the tree
        """
        self.q_start = Node(q_start)
        self.q_goal = Node(q_goal)
        self.obstacles = obstacles
        self.robot_radius = robot_radius
        self.q_limits = q_limits if q_limits else [(TARGET_X_MIN, TARGET_X_MAX), 
                                                     (TARGET_Y_MIN, TARGET_Y_MAX)]
        self.max_iter = max_iter
        self.step_size = step_size
        self.node_list = [self.q_start]
        self.goal_sample_rate = 0.1  # 10% goal biasing

    def check_node_collision(self, position):
        """
        Check if a position collides with any obstacle
        
        Args:
            position: [x, y] position to check
            
        Returns:
            bool: True if collision detected, False otherwise
        """
        pos = np.array(position)
        for (ox, oy, radius) in self.obstacles:
            distance = np.linalg.norm(pos - np.array([ox, oy]))
            if distance <= (radius + self.robot_radius):
                return True
        return False

    def check_edge_collision(self, position_start, position_end, discretization_step=0.01):
        """
        Check for collision along the edge between two positions
        
        Args:
            position_start: Starting [x, y] position
            position_end: Ending [x, y] position
            discretization_step: Maximum interpolation distance before collision check
            
        Returns:
            bool: True if collision detected, False otherwise
        """
        start = np.array(position_start)
        end = np.array(position_end)
        
        # Calculate distance
        distance = np.linalg.norm(end - start)
        
        # Calculate number of steps
        num_steps = int(np.ceil(distance / discretization_step))
        
        # If distance is very small, just check end point
        if num_steps == 0:
            return self.check_node_collision(position_end)
        
        # Interpolate and check collision at each point
        for i in range(num_steps + 1):
            t = i / num_steps
            interpolated_position = start + t * (end - start)
            
            if self.check_node_collision(interpolated_position):
                return True
        
        return False

    def step(self, from_node, to_position):
        """
        Step from from_node towards to_position
        Returns new position within step_size distance
        """
        from_pos = np.array(from_node.position)
        to_pos = np.array(to_position)
        
        # Calculate direction and distance
        direction = to_pos - from_pos
        distance = np.linalg.norm(direction)
        
        # If within step_size, return the target
        if distance <= self.step_size:
            return to_pos
        
        # Otherwise, step in the direction with step_size distance
        normalized_direction = direction / distance
        new_position = from_pos + normalized_direction * self.step_size
        
        return new_position

    def get_nearest_node(self, random_point):
        """Find the nearest node in the tree to a given point"""
        random_point = np.array(random_point)
        
        # Calculate distances to all nodes
        distances = [np.linalg.norm(node.position - random_point) for node in self.node_list]
        
        # Find the index of the minimum distance
        nearest_index = np.argmin(distances)
        
        return self.node_list[nearest_index]

    def plan(self):
        """
        Run the RRT algorithm to find a path
        
        Returns:
            np.array: Nx2 path array from start to goal, or None if no path found
        """
        for iteration in range(self.max_iter):
            # Sample random point with goal biasing
            if random.random() < self.goal_sample_rate:
                random_point = self.q_goal.position
            else:
                # Sample random point within workspace bounds
                random_point = np.array([
                    random.uniform(limit[0], limit[1]) for limit in self.q_limits
                ])
            
            # Find nearest node in tree
            nearest_node = self.get_nearest_node(random_point)
            
            # Step towards random point
            new_position = self.step(nearest_node, random_point)
            
            # Check if the edge is collision-free
            if not self.check_edge_collision(nearest_node.position, new_position):
                # Create new node
                new_node = Node(new_position)
                new_node.parent = nearest_node
                new_node.cost = nearest_node.cost + np.linalg.norm(new_position - nearest_node.position)
                self.node_list.append(new_node)
                
                # Check if goal is reached
                distance_to_goal = np.linalg.norm(new_position - self.q_goal.position)
                if distance_to_goal < self.step_size:
                    # Try to connect to goal
                    if not self.check_edge_collision(new_position, self.q_goal.position):
                        # Goal reached! Extract path
                        self.q_goal.parent = new_node
                        self.q_goal.cost = new_node.cost + distance_to_goal
                        path = self.extract_path()
                        print(f"  Path found in {iteration + 1} iterations!")
                        return path
        
        print(f"  Max iterations ({self.max_iter}) reached without finding a path.")
        return None
    
    def extract_path(self):
        """Extract the path from start to goal by backtracking through parents"""
        path = []
        current_node = self.q_goal
        
        while current_node is not None:
            path.append(current_node.position.tolist())
            current_node = current_node.parent
        
        # Reverse to get path from start to goal
        path.reverse()
        
        return path


def get_object_obstacles_for_rrt(env_idx, object_states_tensor):
    """Extract obstacle coordinates and radii for RRT"""
    if object_states_tensor is None or object_states_tensor.shape[1] == 0:
        return []
    
    obstacles = []
    for obj_idx in range(object_states_tensor.shape[1]):
        obj_pos = object_states_tensor[env_idx, obj_idx, :3].cpu().numpy()
        obstacles.append((obj_pos[0], obj_pos[1], OBJECT_RADIUS))
    
    return obstacles
