from quantbt.ray.cluster_manager import RayClusterManager

ray_cluster_config = {
    "num_cpus": 2,
    "object_store_memory": 1024 * 1024 * 1024,
}

manager = RayClusterManager(ray_cluster_config)
manager.initialize_cluster()

print(manager.cluster_info)