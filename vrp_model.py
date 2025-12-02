"""
MILP Model for Capacitated Vehicle Routing Problem with Time Windows (CVRPTW)

This module implements the Mixed-Integer Linear Programming formulation using Gurobi.
The model includes capacity constraints and SOFT time window constraints.

Mathematical Formulation:
------------------------
Decision Variables:
    a[i,j] \in {0,1}: Binary variable = 1 if vehicle travels from node i to node j, i \neq j
    s[i] >= 0: Continuous variable representing service start time at customer i
    y[i] >= 0: Continuous variable representing cumulative load before serving customer i
    slack[i] >= 0: Slack variable for time window violations (lateness beyond preferred window) for customer i

Objective:
    Minimize: (sum_i) (sum_j \neq i) distance[i,j] * x[i,j] + penalty_weight * (Sum_i) slack[i]
    (total travel distance + penalized time window violations)

Constraints:
    1. Fleet size: Number of vehicles leaving depot <= V
    2. Flow balance: Vehicles leaving depot = vehicles returning to depot
    3. Visit once: Each customer visited exactly once (inflow = outflow = 1)
    4. Time sequencing: s[j] >= s[i] + travel_time[i,j]*x[i,j] + delta - M(1-x[i,j])
    5. Load tracking: y[j] >= y[i] + demand[i]*x[i,j] - M(1-x[i,j])
    6. Capacity limit: y[i] <= vehicle_capacity for all i
    7. Earliest time: s[i] >= earliest[i] for all customers (hard constraint - cannot arrive early)
    8. Latest time with slack: s[i] <= latest[i] + slack[i] for all customers (soft constraint)
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np


class CVRPTWModel:
    """
    Gurobi-based MILP model for CVRPTW.
    """

    def __init__(self, data_generator):
        """
        Initialize and build the MILP model.

        Args:
            data_generator: CVRPTWDataGenerator object containing problem data
        """
        self.data = data_generator
        self.config = data_generator.config
        self.n = data_generator.get_num_nodes()  # Total nodes (depot + customers)

        # Constant for constraint deactivation
        self.BIG_M = 1e6

        # Create Gurobi model
        self.model = gp.Model("CVRPTW")

        # Build the model
        self._create_variables()
        self._add_constraints()
        self._set_objective()

        # Solution storage
        self.routes = []
        self.solution = {}

    def _create_variables(self):
        """
        Create decision variables for the MILP model.
        """
        print("Creating decision variables...")

        # a[i,j]: Binary routing variables
        # 1 if vehicle travels from node i to node j, 0 otherwise
        self.a = self.model.addVars(
            self.n, self.n,
            vtype=GRB.BINARY,
            name="x"
        )

        # s[i]: Service start time at customer i
        self.s = self.model.addVars(
            self.n,
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            name="s"
        )

        # y[i]: Cumulative load of vehicle before serving customer i
        self.y = self.model.addVars(
            self.n,
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            name="y"
        )

        # slack[i]: Time window slack variable (how late service is beyond preferred window)
        # This is a soft constraint variable - penalized in objective
        self.slack = self.model.addVars(
            self.n,
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            name="slack"
        )

        self.model.update()

    def _add_constraints(self):
        """
        Add all constraints to the MILP model.
        """
        print("Adding constraints...")

        # --- Constraint 1: Fleet size limit ---
        # At most V vehicles can leave the depot
        self.model.addConstr(
            gp.quicksum(self.a[0, j] for j in range(1, self.n)) <= self.config['num_vehicles'],
            name="fleet_size"
        )

        # --- Constraint 2: Flow balance at depot ---
        # Number of vehicles leaving depot = number returning
        self.model.addConstr(
            gp.quicksum(self.a[0, j] for j in range(1, self.n)) ==
            gp.quicksum(self.a[j, 0] for j in range(1, self.n)),
            name="depot_flow_balance"
        )

        # --- Constraint 3: Each customer visited exactly once ---
        for i in range(1, self.n):
            # Exactly one vehicle arrives at customer i
            self.model.addConstr(
                gp.quicksum(self.a[j, i] for j in range(self.n) if j != i) == 1,
                name=f"arrive_customer_{i}"
            )

            # Exactly one vehicle leaves customer i
            self.model.addConstr(
                gp.quicksum(self.a[i, j] for j in range(self.n) if j != i) == 1,
                name=f"leave_customer_{i}"
            )

        # --- Constraint 4: Time window sequencing ---
        # If vehicle goes from i to j, then service at j starts after
        # finishing service at i plus travel time
        # s[j] >= s[i] + travel_time[i,j] + service_duration - M(1-a[i,j])
        service_duration = self.config['service_duration_hours']

        for i in range(self.n):
            for j in range(1, self.n):  # j cannot be depot for service time
                if i != j:
                    self.model.addConstr(
                        self.s[j] >= self.s[i] +
                        self.data.travel_time_matrix[i, j] * self.a[i, j] +
                        service_duration -
                        self.BIG_M * (1 - self.a[i, j]),
                        name=f"time_sequence_{i}_{j}"
                    )

        # --- Constraint 5: Capacity tracking  ---
        # If vehicle goes from i to j, load at j accounts for demand served at i
        for i in range(1, self.n):  # Start from customers (not depot)
            for j in range(1, self.n):
                if i != j:
                    self.model.addConstr(
                        self.y[j] >= self.y[i] +
                        self.data.demands[i] * self.a[i, j] -
                        self.BIG_M * (1 - self.a[i, j]),
                        name=f"load_tracking_{i}_{j}"
                    )

        # --- Constraint 6: Vehicle capacity limits ---
        # y[i] represents load BEFORE serving customer i
        # Total load including customer i's demand must not exceed capacity
        for i in range(self.n):
            self.model.addConstr(
                self.y[i] + self.data.demands[i] <= self.config['vehicle_capacity'],
                name=f"capacity_limit_{i}"
            )

        # --- Constraint 7: Time window constraints (SOFT) ---
        # Hard constraint: Cannot arrive before earliest time
        for i in range(1, self.n):
            self.model.addConstr(
                self.s[i] >= self.data.earliest_times[i],
                name=f"earliest_time_{i}"
            )

        # Soft constraint: Can arrive late, but with slack penalty
        # s[i] <= latest[i] + slack[i]
        # Slack represents how late the service is beyond preferred window
        for i in range(1, self.n):
            self.model.addConstr(
                self.s[i] <= self.data.latest_times[i] + self.slack[i],
                name=f"latest_time_with_slack_{i}"
            )

        # Depot service time fixed at day start
        self.model.addConstr(
            self.s[0] == self.config['day_start_hour'],
            name="depot_start_time"
        )

        self.model.update()

    def _set_objective(self):
        """
        Set the objective function: minimize total travel distance + penalized time window violations.
        """
        print("Setting objective function...")

        # Get penalty weight from config (defaults to high value to strongly discourage lateness)
        penalty_weight = self.config.get('time_window_penalty', 1000.0)

        # Distance component
        distance_cost = gp.quicksum(
            self.data.distance_matrix[i, j] * self.a[i, j]
            for i in range(self.n)
            for j in range(self.n)
            if i != j
        )

        # Time window violation penalty (sum of all slack variables)
        time_window_penalty = penalty_weight * gp.quicksum(
            self.slack[i] for i in range(1, self.n)  # Only customers, not depot
        )

        # Combined objective
        objective = distance_cost + time_window_penalty

        self.model.setObjective(objective, GRB.MINIMIZE)
        self.model.update()

    def solve(self, time_limit=300, verbose=True):
        """
        Solve the MILP model.

        Args:
            time_limit (int): Time limit in seconds for solver
            verbose (bool): Whether to print solver output

        Returns:
            bool: True if optimal solution found, False otherwise
        """
        print("\n" + "=" * 60)
        print("Solving CVRPTW Model...")
        print("=" * 60)

        # Set solver parameters
        self.model.setParam('TimeLimit', time_limit)
        if not verbose:
            self.model.setParam('OutputFlag', 0)

        # Optimize
        self.model.optimize()

        # Check solution status
        if self.model.status == GRB.OPTIMAL:
            print("\nSuccess!!! Optimal solution found!")
            self._extract_solution()
            return True
        elif self.model.status == GRB.TIME_LIMIT:
            print("\nStopping!!! Time limit reached. Best solution found so far:")
            self._extract_solution()
            return True
        else:
            print(f"\nError!!! Optimization failed with status: {self.model.status}")
            return False

    def _extract_solution(self):
        """
        Extract solution from the optimized model.
        """
        # Extract routing decisions
        x_solution = np.zeros((self.n, self.n), dtype=int)
        for i in range(self.n):
            for j in range(self.n):
                if i != j and self.a[i, j].X > 0.5:  # Binary variable threshold
                    x_solution[i, j] = 1

        # Extract service times
        s_solution = np.array([self.s[i].X for i in range(self.n)])

        # Extract loads
        y_solution = np.array([self.y[i].X for i in range(self.n)])

        # Extract slack (time window violations)
        slack_solution = np.array([self.slack[i].X for i in range(self.n)])

        # Store solution
        self.solution = {
            'x': x_solution,
            's': s_solution,
            'y': y_solution,
            'slack': slack_solution,
            'objective_value': self.model.ObjVal
        }

        # Extract routes
        self.routes = self._trace_routes(x_solution)

    def _trace_routes(self, x_matrix):
        """
        Trace individual vehicle routes from the routing decision matrix.

        Args:
            x_matrix (np.array): Binary matrix where x[i,j]=1 means arc (i,j) is used

        Returns:
            list: List of routes, where each route is a list of node indices
        """
        routes = []

        # Find all vehicles leaving depot
        for j in range(1, self.n):
            if x_matrix[0, j] == 1:
                route = [0, j]  # Start from depot, go to first customer

                # Trace the route until returning to depot
                current = j
                while True:
                    # Find next node
                    next_nodes = np.where(x_matrix[current] == 1)[0]
                    if len(next_nodes) == 0:
                        break

                    next_node = next_nodes[0]
                    route.append(next_node)

                    if next_node == 0:  # Returned to depot
                        break

                    current = next_node

                routes.append(route)

        return routes

    def print_solution(self):
        """
        Print detailed solution information.
        """
        if not self.solution:
            print("No solution available. Run solve() first.")
            return

        # Calculate total time window violations
        total_slack = np.sum(self.solution['slack'])
        num_violations = np.sum(self.solution['slack'] > 0.01)  # Count customers with slack > 0

        print("\n" + "=" * 60)
        print("SOLUTION SUMMARY")
        print("=" * 60)
        print(f"Total objective value: {self.solution['objective_value']:.2f}")
        print(f"Number of vehicles used: {len(self.routes)}")
        print(f"Total time window violations: {total_slack:.2f} hours")
        print(f"Customers with delayed service: {num_violations} out of {self.n - 1}")
        print("\n" + "-" * 60)
        print("Vehicle Routes:")
        print("-" * 60)

        for idx, route in enumerate(self.routes, 1):
            route_distance = sum(
                self.data.distance_matrix[route[i], route[i+1]]
                for i in range(len(route) - 1)
            )

            # Calculate route load
            route_load = sum(self.data.demands[node] for node in route)

            print(f"\nVehicle {idx}:")
            print(f"  Route: {' -> '.join(map(str, route))}")
            print(f"  Distance: {route_distance:.2f} miles")
            print(f"  Total load: {route_load} units")

            # Print timing for each customer on route
            print(f"  Schedule:")
            for node in route:
                if node == 0:
                    print(f"    Node {node} (Depot): Depart at {self.solution['s'][node]:.2f}h")
                else:
                    slack = self.solution['slack'][node]
                    if slack > 0.01:  # Significant slack
                        print(f"    Node {node}: Service at {self.solution['s'][node]:.2f}h "
                              f"(window: {self.data.earliest_times[node]:.1f}-{self.data.latest_times[node]:.1f}h) "
                              f"[LATE by {slack:.2f}h]")
                    else:
                        print(f"    Node {node}: Service at {self.solution['s'][node]:.2f}h "
                              f"(window: {self.data.earliest_times[node]:.1f}-{self.data.latest_times[node]:.1f}h) "
                              f"[ON TIME]")

        print("\n" + "=" * 60)

    def export_model(self, filename="cvrptw_model.lp"):
        """
        Export the model to a .lp file for inspection.

        Args:
            filename (str): Output filename
        """
        filepath = f"{filename}"
        self.model.write(filepath)
        print(f"\nModel exported to: {filepath}")

    def save_solution(self, filename="solution.pkl"):
        """
        Save the solution and data to a pickle file for later use.
        This allows regenerating visualizations without re-solving.

        Args:
            filename (str): Output filename for the saved solution
        """
        import pickle

        if not self.solution:
            print("No solution to save. Run solve() first.")
            return

        save_data = {
            'solution': self.solution,
            'routes': self.routes,
            'config': self.config,
            'data': {
                'locations': self.data.locations,
                'demands': self.data.demands,
                'earliest_times': self.data.earliest_times,
                'latest_times': self.data.latest_times,
                'distance_matrix': self.data.distance_matrix,
                'travel_time_matrix': self.data.travel_time_matrix
            }
        }

        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"\nSuccess!!! Solution saved to: {filename}")

    @classmethod
    def load_solution(cls, filename="solution.pkl"):
        """
        Load a previously saved solution from a pickle file.
        Returns a dictionary with solution data and a reconstructed data generator.

        Args:
            filename (str): Input filename for the saved solution

        Returns:
            tuple: (model_with_solution, data_generator)
        """
        import pickle
        from data_generator import CVRPTWDataGenerator

        with open(filename, 'rb') as f:
            save_data = pickle.load(f)

        # Reconstruct data generator
        data_generator = CVRPTWDataGenerator(save_data['config'])
        data_generator.locations = save_data['data']['locations']
        data_generator.demands = save_data['data']['demands']
        data_generator.earliest_times = save_data['data']['earliest_times']
        data_generator.latest_times = save_data['data']['latest_times']
        data_generator.distance_matrix = save_data['data']['distance_matrix']
        data_generator.travel_time_matrix = save_data['data']['travel_time_matrix']

        # Create a model instance (without solving)
        model = cls(data_generator)
        model.solution = save_data['solution']
        model.routes = save_data['routes']

        print(f"\nSuccess!!! Solution loaded from: {filename}")
        return model, data_generator
