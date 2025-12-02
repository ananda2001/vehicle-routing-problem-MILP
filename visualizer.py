"""
Visualization Module for CVRPTW Solutions

This module provides functions to visualize the vehicle routing problem
on geographic maps with real basemap tiles (OpenStreetMap).
"""

import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx
from matplotlib.cm import get_cmap
import numpy as np


class RouteVisualizer:
    """
    Visualizes CVRPTW problem instances and solutions on geographic maps.
    """

    def __init__(self, data_generator):
        """
        Initialize visualizer with problem data.

        Args:
            data_generator: CVRPTWDataGenerator object
        """
        self.data = data_generator
        self.config = data_generator.config

    def plot_network(self, save_path="network.png"):
        """
        Plot the network showing depot and customer locations.

        Args:
            save_path (str): Path to save the figure
        """
        print(f"\nGenerating network visualization...")

        fig, ax = plt.subplots(figsize=(12, 10), dpi=150)

        # Create GeoDataFrame for locations
        geometries = [Point(lon, lat) for lat, lon in self.data.locations]
        labels = ['Depot'] + [f'C{i}' for i in range(1, len(self.data.locations))]

        gdf = gpd.GeoDataFrame({
            'label': labels,
            'demand': self.data.demands
        }, geometry=geometries, crs='EPSG:4326')

        # Separate depot and customers
        depot = gdf[gdf['label'] == 'Depot']
        customers = gdf[gdf['label'] != 'Depot']

        # Plot depot
        depot.plot(
            ax=ax,
            markersize=300,
            color='red',
            marker='*',
            label='Depot (Columbus, OH)',
            edgecolors='black',
            linewidth=2,
            zorder=5
        )

        # Plot customers
        customers.plot(
            ax=ax,
            markersize=120,
            color='blue',
            marker='o',
            label='Customers',
            edgecolors='black',
            linewidth=1,
            alpha=0.8,
            zorder=4
        )

        # Add basemap (OpenStreetMap tiles)
        ctx.add_basemap(
            ax=ax,
            crs=gdf.crs,
            source=ctx.providers.OpenStreetMap.Mapnik,
            attribution=""
        )

        # Styling
        ax.set_title(
            f'CVRPTW Network - Columbus, Ohio Area\n'
            f'{self.config["num_customers"]} Customers, '
            f'{self.config["num_vehicles"]} Vehicles',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
        ax.set_xlabel('Longitude', fontsize=11)
        ax.set_ylabel('Latitude', fontsize=11)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Network plot saved to: {save_path}")
        plt.close()

    def plot_solution(self, model, save_path="solution.png"):
        """
        Plot the optimized routes on a geographic map.
        Customers with time window violations are highlighted in red.

        Args:
            model: CVRPTWModel object with solved solution
            save_path (str): Path to save the figure
        """
        if not model.routes:
            print("No routes to plot. Solve the model first.")
            return

        print(f"\nGenerating solution visualization...")

        fig, ax = plt.subplots(figsize=(12, 10), dpi=150)

        # Create GeoDataFrame for locations
        geometries = [Point(lon, lat) for lat, lon in self.data.locations]
        labels = ['Depot'] + [f'C{i}' for i in range(1, len(self.data.locations))]

        gdf = gpd.GeoDataFrame({
            'label': labels,
            'demand': self.data.demands
        }, geometry=geometries, crs='EPSG:4326')

        # Generate colors for routes
        num_routes = len(model.routes)
        cmap = get_cmap('tab10', num_routes)
        colors = [cmap(i) for i in range(num_routes)]

        # Plot routes
        for route_idx, route in enumerate(model.routes):
            route_coords = [self.data.locations[node] for node in route]
            # Convert to (lon, lat) for plotting
            route_lons = [coord[1] for coord in route_coords]
            route_lats = [coord[0] for coord in route_coords]

            # Plot route line segments with arrows
            for i in range(len(route_lons) - 1):
                start_lon, start_lat = route_lons[i], route_lats[i]
                end_lon, end_lat = route_lons[i + 1], route_lats[i + 1]

                # Plot line
                ax.plot(
                    [start_lon, end_lon],
                    [start_lat, end_lat],
                    color=colors[route_idx],
                    linewidth=2.5,
                    alpha=0.7,
                    zorder=2,
                    label=f'Vehicle {route_idx + 1}' if i == 0 else ""
                )

                # Add directional arrow
                ax.annotate(
                    '',
                    xy=(end_lon, end_lat),
                    xytext=(start_lon, start_lat),
                    arrowprops=dict(
                        arrowstyle='->',
                        color=colors[route_idx],
                        lw=2,
                        alpha=0.8
                    ),
                    zorder=3
                )

        # Plot depot
        depot = gdf[gdf['label'] == 'Depot']
        depot.plot(
            ax=ax,
            markersize=300,
            color='red',
            marker='*',
            edgecolors='black',
            linewidth=2,
            zorder=5
        )

        # Separate customers by time window violation status
        # Get slack values for all customers
        slack_values = model.solution['slack']

        # Create lists for on-time and late customers
        ontime_geoms = []
        late_geoms = []

        for i in range(1, len(self.data.locations)):
            if slack_values[i] > 0.01:  # Late
                late_geoms.append(geometries[i])
            else:  # On time
                ontime_geoms.append(geometries[i])

        # Plot on-time customers (green)
        if ontime_geoms:
            ontime_gdf = gpd.GeoDataFrame(geometry=ontime_geoms, crs='EPSG:4326')
            ontime_gdf.plot(
                ax=ax,
                markersize=120,
                color='lightgreen',
                marker='o',
                edgecolors='darkgreen',
                linewidth=1.5,
                alpha=0.9,
                zorder=4,
                label='On-time customers'
            )

        # Plot late customers (orange/red)
        if late_geoms:
            late_gdf = gpd.GeoDataFrame(geometry=late_geoms, crs='EPSG:4326')
            late_gdf.plot(
                ax=ax,
                markersize=120,
                color='orange',
                marker='o',
                edgecolors='darkred',
                linewidth=1.5,
                alpha=0.9,
                zorder=4,
                label='Late customers'
            )

        # Add basemap
        ctx.add_basemap(
            ax=ax,
            crs=gdf.crs,
            source=ctx.providers.OpenStreetMap.Mapnik,
            attribution=""
        )

        # Styling
        total_slack = np.sum(model.solution['slack'])
        num_late = len(late_geoms)

        ax.set_title(
            f'CVRPTW Optimal Solution - Columbus, Ohio Area\n'
            f'{num_routes} Vehicle(s) | '
            f'Time Window Violations: {total_slack:.2f}h ({num_late} customers)',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax.set_xlabel('Longitude', fontsize=11)
        ax.set_ylabel('Latitude', fontsize=11)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Solution plot saved to: {save_path}")
        plt.close()

    def plot_both(self, model=None, network_path="network.png",
                  solution_path="solution.png"):
        """
        Plot both network and solution side by side.

        Args:
            model: CVRPTWModel object (optional, if None only plots network)
            network_path (str): Path to save network figure
            solution_path (str): Path to save solution figure
        """
        self.plot_network(save_path=network_path)

        if model and model.routes:
            self.plot_solution(model, save_path=solution_path)

    def plot_schedule_timeline(self, model, save_path="schedule_timeline.png"):
        """
        Plot time window adherence for each vehicle route.

        Creates a timeline plot for each vehicle showing:
        - Time windows as vertical bars/brackets
        - Actual service times as dots (blue=on time, red=late)
        - Lines connecting service times to show route progression

        Args:
            model: CVRPTWModel object with solved solution
            save_path (str): Path to save the figure
        """
        if not model.routes:
            print("No routes to plot. Solve the model first.")
            return

        print(f"\nGenerating schedule timeline visualization...")

        # Disable LaTeX rendering (simpler, more compatible)
        use_latex = False

        num_vehicles = len(model.routes)

        # Create subplots - one per vehicle
        fig, axes = plt.subplots(num_vehicles, 1, figsize=(14, 4 * num_vehicles), dpi=150)

        # Handle case of single vehicle
        if num_vehicles == 1:
            axes = [axes]

        # Plot each vehicle's schedule
        for vehicle_idx, (ax, route) in enumerate(zip(axes, model.routes), 1):
            # Get customers in this route (exclude depot)
            customers = [node for node in route if node != 0]

            if not customers:
                continue

            num_customers = len(customers)
            x_positions = np.arange(num_customers)

            # Plot time windows as error bars (vertical brackets)
            earliest = [self.data.earliest_times[c] for c in customers]
            latest = [self.data.latest_times[c] for c in customers]
            window_centers = [(e + l) / 2 for e, l in zip(earliest, latest)]
            window_heights = [l - e for e, l in zip(earliest, latest)]

            # Draw time windows as gray bars
            ax.bar(x_positions, window_heights, bottom=earliest,
                   alpha=0.3, color='lightgray', width=0.6,
                   label='Time Window', edgecolor='gray', linewidth=1.5)

            # Get actual service times
            service_times = [model.solution['s'][c] for c in customers]
            slack_values = [model.solution['slack'][c] for c in customers]

            # Separate on-time and late customers
            ontime_indices = [i for i, s in enumerate(slack_values) if s <= 0.01]
            late_indices = [i for i, s in enumerate(slack_values) if s > 0.01]

            # Plot on-time service points (blue)
            if ontime_indices:
                ax.scatter([x_positions[i] for i in ontime_indices],
                          [service_times[i] for i in ontime_indices],
                          color='blue', s=150, zorder=5,
                          label='On-time Service', edgecolors='darkblue', linewidth=2)

            # Plot late service points (red) - use 'o' marker (dot) instead of 'X'
            if late_indices:
                ax.scatter([x_positions[i] for i in late_indices],
                          [service_times[i] for i in late_indices],
                          color='red', s=150, zorder=5,
                          label='Late Service', edgecolors='darkred', linewidth=2)

            # Connect service times with lines to show progression
            ax.plot(x_positions, service_times,
                   color='black', linewidth=2, alpha=0.6,
                   linestyle='--', zorder=3, label='Route Progression')

            # Styling
            # Only show x-axis label on the bottom subplot
            if vehicle_idx == num_vehicles:  # Last vehicle (bottom plot)
                if use_latex:
                    ax.set_xlabel(r'\textbf{Customers}', fontsize=14)
                else:
                    ax.set_xlabel('Customers', fontsize=14, fontweight='bold')
            else:
                ax.set_xlabel('')  # No label for upper plots

            # Y-axis label for all subplots
            if use_latex:
                ax.set_ylabel(r'\textbf{Time (hours)}', fontsize=14)
                ax.set_title(rf'\textbf{{Vehicle {vehicle_idx}}}', fontsize=14, pad=10)
            else:
                ax.set_ylabel('Time (hours)', fontsize=14, fontweight='bold')
                ax.set_title(f'Vehicle {vehicle_idx}', fontsize=14, fontweight='bold', pad=10)

            # Remove x-axis tick labels (no customer numbering)
            ax.set_xticks(x_positions)
            ax.set_xticklabels([])

            # Set y-axis to show hours with larger font
            ax.set_ylim(self.config['day_start_hour'] - 0.5,
                       self.config['day_end_hour'] + 0.5)
            ax.tick_params(axis='y', labelsize=14)
            ax.grid(True, axis='y', alpha=0.3, linestyle=':')

            # Add legend with larger font
            ax.legend(loc='upper left', fontsize=11, framealpha=0.9)

            # Add route info as text
            route_distance = sum(
                self.data.distance_matrix[route[i], route[i+1]]
                for i in range(len(route) - 1)
            )
            route_load = sum(self.data.demands[node] for node in customers)
            total_slack = sum(slack_values)

            info_text = f'Distance: {route_distance:.1f} mi | Load: {route_load} units | Delay: {total_slack:.2f} h'
            ax.text(0.98, 0.98, info_text,
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Success!!! Schedule timeline plot saved to: {save_path}")
        plt.close()
